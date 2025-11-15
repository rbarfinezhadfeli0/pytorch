# Documentation: `docs/torch/optim/lr_scheduler.py_docs.md`

## File Metadata

- **Path**: `docs/torch/optim/lr_scheduler.py_docs.md`
- **Size**: 54,201 bytes (52.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/optim/lr_scheduler.py`

## File Metadata

- **Path**: `torch/optim/lr_scheduler.py`
- **Size**: 103,794 bytes (101.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
r"""Learning Rate Scheduler."""

from __future__ import annotations

import math
import types
import warnings
from bisect import bisect_right
from collections import Counter
from functools import partial, wraps
from typing import (
    Any,
    cast,
    Literal,
    Optional,
    SupportsFloat,
    TYPE_CHECKING,
    TypedDict,
    Union,
)
from typing_extensions import override, Self
from weakref import ref

from torch import inf, Tensor

from .optimizer import _to_scalar, Optimizer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


__all__ = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "SequentialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "ReduceLROnPlateau",
    "CyclicLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialLR",
    "LRScheduler",
]

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


def _format_param(name: str, optimizer: Optimizer, param):
    """Return correctly formatted lr/momentum for each param group."""

    def _copy(_param):
        return _param.clone() if isinstance(_param, Tensor) else _param

    if isinstance(param, (list, tuple)):
        if len(param) != len(optimizer.param_groups):
            raise ValueError(
                f"{name} must have the same length as optimizer.param_groups. "
                f"{name} has {len(param)} values, param_groups has {len(optimizer.param_groups)}."
            )
    else:
        param = [param] * len(optimizer.param_groups)

    return list(map(_copy, param))


def _param_groups_val_list(optimizer: Optimizer, key: str) -> list[Any]:
    """Create a list containing group[key] for each optimizer param_group.
    Prevents aliasing when group[key] could be a Tensor.
    Raises a KeyError when group[key] does not exist.
    """
    return [
        group[key].clone() if isinstance(group[key], Tensor) else group[key]
        for group in optimizer.param_groups
    ]


def _update_param_group_val(
    param_group: dict[str, Any], key: str, val: float | Tensor
) -> None:
    """Set param_group[key] to val without aliasing or assignment when they're
    both tensors. Raises a KeyError if param_group[key] does not exist.
    """
    if isinstance(param_group[key], Tensor):
        param_group[key].fill_(_to_scalar(val))
    else:
        param_group[key] = val


class LRScheduler:
    r"""Base class for all learning rate schedulers.

    Subclasses implement :meth:`get_lr` and optionally override :meth:`step` to
    define scheduling behavior.

    Args:
        optimizer (Optimizer): The optimizer this scheduler will adjust the
            learning rates of.
        last_epoch (int): Index of the last epoch seen by the scheduler. Use
            ``-1`` (default) to initialize the scheduler. Only use a non-default
            value when restoring this scheduler from a saved checkpoint.

    .. warning::
        Initializing a scheduler overwrites its optimizer's
        ``param_group["lr"]``\s. When restoring a checkpoint, initialize the
        scheduler **before** calling your optimizer's
        :meth:`~torch.optim.Optimizer.load_state_dict` to avoid overwriting the
        loaded learning rates.
    """

    _get_lr_called_within_step: bool = False
    _is_initial: bool = False

    def __init__(
        self,
        optimizer: Optimizer,
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                initial_lr = group["lr"]
                if isinstance(initial_lr, Tensor):
                    initial_lr = initial_lr.clone()
                group.setdefault("initial_lr", initial_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        f"param 'initial_lr' is not specified in param_groups[{i}] when resuming scheduler with last_epoch >= 0.\n"
                        "This typically happens when:\n"
                        "1. You're trying to resume training from a checkpoint but haven't properly loaded the optimizer state\n"
                        "2. You're using last_epoch >= 0 for a fresh training run (not recommended)"
                    )
        self.base_lrs: list[float | Tensor] = _param_groups_val_list(
            optimizer, "initial_lr"
        )
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def patch_track_step_called(opt: Optimizer):
            if hasattr(opt.step, "_wrapped_by_lr_sched"):
                # we've already patched
                return opt.step

            def wrap_step(step_fn):
                opt_ref = ref(self.optimizer)
                func = step_fn.__func__

                @wraps(func)
                def wrapper(*args, **kwargs):
                    opt = opt_ref()
                    opt._opt_called = True  # type: ignore[union-attr]
                    return func.__get__(opt, opt.__class__)(*args, **kwargs)

                wrapper._wrapped_by_lr_sched = True  # type: ignore[attr-defined]
                return wrapper

            opt.step = wrap_step(opt.step)  # type: ignore[method-assign]

        patch_track_step_called(self.optimizer)
        self._initial_step()

    def _initial_step(self) -> None:
        """Initialize step counts and perform a step."""
        self._step_count = 0
        with _initial_mode(self):
            self.step()

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> list[float | Tensor]:
        r"""Get the most recent learning rates computed by this scheduler.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates with entries
            for each of the optimizer's
            :attr:`~torch.optim.Optimizer.param_groups`, with the same types as
            their ``group["lr"]``\s.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        # We always update self._last_lr with _param_groups_val_list, so it's a
        # .clone() of the group["lr"]s. If we didn't do this, the user could
        # corrupt their learning rates by modifying the outputs in place.
        return self._last_lr

    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> None:
        """Step the scheduler.

        Args:
            epoch (int, optional):
                .. deprecated:: 1.4
                    If provided, sets :attr:`last_epoch` to ``epoch`` and uses
                    :meth:`_get_closed_form_lr` if it is available. This is not
                    universally supported. Use :meth:`step` without arguments
                    instead.

        .. note::
            Call this method after calling the optimizer's
            :meth:`~torch.optim.Optimizer.step`.
        """
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_wrapped_by_lr_sched"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                    stacklevel=2,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif not getattr(self.optimizer, "_opt_called", False):
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                    stacklevel=2,
                )

        self._step_count += 1
        if epoch is not None:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning, stacklevel=2)
        self._update_lr(epoch)

    def _update_lr(self, epoch: Optional[int] = None) -> None:
        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = cast(
                        list[Union[float, Tensor]], self._get_closed_form_lr()
                    )
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values, strict=True):
            _update_param_group_val(param_group, "lr", lr)

        self._last_lr: list[float | Tensor] = _param_groups_val_list(
            self.optimizer, "lr"
        )


def _warn_get_lr_called_within_step(lr_scheduler: LRScheduler) -> None:
    if not lr_scheduler._get_lr_called_within_step:
        warnings.warn(
            "To get the last learning rate computed by the scheduler, "
            "please use `get_last_lr()`.",
            UserWarning,
            stacklevel=2,
        )


# Including _LRScheduler for backwards compatibility
# Subclass instead of assign because we want __name__ of _LRScheduler to be _LRScheduler (assigning would make it LRScheduler).
class _LRScheduler(LRScheduler):
    pass


class _enable_get_lr_call:
    def __init__(self, o: LRScheduler) -> None:
        self.o = o

    def __enter__(self) -> Self:
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.o._get_lr_called_within_step = False


class _initial_mode:
    def __init__(self, o: LRScheduler) -> None:
        self.o = o

    def __enter__(self):
        self.o._is_initial = True

    def __exit__(self, type, value, traceback):
        self.o._is_initial = False


class LambdaLR(LRScheduler):
    """Sets the initial learning rate.

    The learning rate of each parameter group is set to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer has two groups.
        >>> num_epochs = 100
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95**epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
        >>>
        >>> # Alternatively, you can use a single lambda function for all groups.
        >>> scheduler = LambdaLR(opt, lr_lambda=lambda epoch: epoch // 30)
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/LambdaLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Union[Callable[[int], float], list[Callable[[int], float]]],
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        self.optimizer = optimizer

        self.lr_lambdas: list[Callable[[int], float]]
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch)

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                # pyrefly: ignore [unsupported-operation]
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["lr_lambdas"] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the :attr:`base_lrs` by the outputs of the :attr:`lr_lambdas` at
        :attr:`last_epoch`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        return [
            base_lr * lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs, strict=True)
        ]


class MultiplicativeLR(LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given in the specified function.

    When last_epoch=-1, set initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> lmbda = lambda epoch: 0.95
        >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/MultiplicativeLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Union[Callable[[int], float], list[Callable[[int], float]]],
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        self.optimizer = optimizer

        self.lr_lambdas: list[Callable[[int], float]]
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        for lr_lambda in self.lr_lambdas:
            if not callable(lr_lambda):
                raise TypeError(
                    f"lr_lambda should be a function, but got {type(lr_lambda).__name__}"
                )
        super().__init__(optimizer, last_epoch)

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                # pyrefly: ignore [unsupported-operation]
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["lr_lambdas"] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the current ``group["lr"]``\s in each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by the outputs of the
        :attr:`lr_lambdas` at :attr:`last_epoch`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if not self._is_initial:
            return [
                group["lr"] * lmbda(self.last_epoch)
                for lmbda, group in zip(
                    self.lr_lambdas, self.optimizer.param_groups, strict=True
                )
            ]
        else:
            return _param_groups_val_list(self.optimizer, "lr")


class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/StepLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        If the current epoch is a non-zero multiple of :attr:`step_size`, we
        scale the current ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by :attr:`gamma`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return _param_groups_val_list(self.optimizer, "lr")
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]


class MultiStepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/MultiStepLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Iterable[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        If the current epoch is in :attr:`milestones`, decays the
        ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by :attr:`gamma`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.

        .. note::
            If the current epoch appears in :attr:`milestones` ``n`` times, we
            scale by :attr:`gamma` to the power of ``n``
        """
        _warn_get_lr_called_within_step(self)

        if self.last_epoch not in self.milestones:
            return _param_groups_val_list(self.optimizer, "lr")
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        milestones = sorted(self.milestones.elements())
        return [
            base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class ConstantLR(LRScheduler):
    """Multiply the learning rate of each parameter group by a small constant factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such multiplication of the small constant factor can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler multiplies the learning rate by the factor.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # ...
        >>> # lr = 0.05    if epoch >= 40
        >>> scheduler = ConstantLR(optimizer, factor=0.5, total_iters=40)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ConstantLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        if factor > 1.0 or factor < 0:
            raise ValueError(
                "Constant multiplicative factor expected to be between 0 and 1."
            )

        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        When :attr:`last_epoch` is 0, this method scales the ``group["lr"]``\s
        in each of the optimizer's :attr:`~torch.optim.Optimizer.param_groups`
        by :attr:`factor`. Once :attr:`total_iters` is reached, it undoes this,
        scaling by ``1 / factor``.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [group["lr"] * self.factor for group in self.optimizer.param_groups]

        if self.last_epoch != self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        return [
            group["lr"] * (1.0 / self.factor) for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            base_lr
            * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
            for base_lr in self.base_lrs
        ]


class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small multiplicative factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.003687  if epoch == 0
        >>> # lr = 0.004875  if epoch == 1
        >>> # lr = 0.006062  if epoch == 2
        >>> # lr = 0.00725   if epoch == 3
        >>> # ...
        >>> # lr = 0.05      if epoch >= 40
        >>> scheduler = LinearLR(optimizer, start_factor=0.05, total_iters=40)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/LinearLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(
                "Starting multiplicative factor expected to be greater than 0 and less or equal to 1."
            )

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` such that successive steps
        interpolate linearly from :attr:`start_factor` up to :attr:`end_factor`
        across :attr:`total_iters` steps.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [
                group["lr"] * self.start_factor for group in self.optimizer.param_groups
            ]

        if self._is_initial or self.last_epoch > self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for base_lr in self.base_lrs
        ]


class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ExponentialLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Multiplies the current ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by :attr:`gamma`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        # when loading from a checkpoint, we don't want _initial_step (called from the constructor)
        # to update the lr one more step ahead of itself.
        if self._is_initial:
            return _param_groups_val_list(self.optimizer, "lr")
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]


class SequentialLR(LRScheduler):
    """Contains a list of schedulers expected to be called sequentially during the optimization process.

    Specifically, the schedulers will be called according to the milestone points, which should provide exact
    intervals by which each scheduler should be called at a given epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.005     if epoch == 0
        >>> # lr = 0.005     if epoch == 1
        >>> # lr = 0.005     if epoch == 2
        >>> # ...
        >>> # lr = 0.05      if epoch == 20
        >>> # lr = 0.045     if epoch == 21
        >>> # lr = 0.0405    if epoch == 22
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=20)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = SequentialLR(
        ...     optimizer,
        ...     schedulers=[scheduler1, scheduler2],
        ...     milestones=[20],
        ... )
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/SequentialLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[LRScheduler],
        milestones: list[int],
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        if len(schedulers) < 1:
            raise ValueError(
                f"{self.__class__.__name__} expects at least one scheduler, but got no scheduler."
            )

        for scheduler_idx, scheduler in enumerate(schedulers):
            if not hasattr(scheduler, "optimizer"):
                raise TypeError(
                    f"{self.__class__.__name__} at index {scheduler_idx} should have `optimizer` as its attribute."
                )
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError(
                    f"{self.__class__.__name__} does not support `ReduceLROnPlateau` scheduler as it "
                    "requires additional kwargs to be specified when calling `step`, "
                    f"but got one at index {scheduler_idx} in the given schedulers sequence."
                )
            if optimizer != scheduler.optimizer:
                raise ValueError(
                    f"{self.__class__.__name__} expects all schedulers to belong to the same optimizer, but "
                    f"got scheduler {scheduler.__class__.__name__} at index {scheduler_idx} has {scheduler.optimizer}, "
                    f"which is different from {optimizer.__class__.__name__}."
                )

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                f"than the number of milestone points, but got number of schedulers {len(schedulers)} and the "
                f"number of milestones to be equal to {len(milestones)}"
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

        # Reset learning rates back to initial values
        for group in self.optimizer.param_groups:
            _update_param_group_val(group, "lr", group["initial_lr"])

        # "Undo" the step performed by other schedulers
        self.recursive_undo()

        # Perform the initial step for only the first scheduler
        self._schedulers[0]._initial_step()

        self._last_lr = schedulers[0].get_last_lr()

    def recursive_undo(self, sched=None) -> None:
        """
        Recursively undo any step performed by the initialisation of
        schedulers.
        """
        scheds = self if sched is None else sched

        if hasattr(scheds, "_schedulers"):
            for s in scheds._schedulers:
                self.recursive_undo(s)
        elif hasattr(scheds, "last_epoch"):
            scheds.last_epoch -= 1

    def step(self) -> None:  # type: ignore[override]
        """Perform a step."""
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler._update_lr(0)
        else:
            scheduler.step()

        self._last_lr = scheduler.get_last_lr()

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        state_dict["_schedulers"] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            # pyrefly: ignore [unsupported-operation]
            state_dict["_schedulers"][idx] = s.state_dict()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["_schedulers"] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


class PolynomialLR(LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function in the given total_iters.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (float): The power of the polynomial. Default: 1.0.

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.0490   if epoch == 0
        >>> # lr = 0.0481   if epoch == 1
        >>> # lr = 0.0472   if epoch == 2
        >>> # ...
        >>> # lr = 0.0      if epoch >= 50
        >>> scheduler = PolynomialLR(optimizer, total_iters=50, power=0.9)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/PolynomialLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` such that the learning rates
        follow

        .. math::
            \texttt{base\_lr} \cdot \left(1 - \frac{\texttt{last\_epoch}}
            {\texttt{total\_iters}} \right)^\texttt{power}

        Returns the current learning rates unchanged after :attr:`total_iters`
        is reached.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if self._is_initial or self.last_epoch > self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        d
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/optim`):

- [`rprop.py_kw.md_docs.md`](./rprop.py_kw.md_docs.md)
- [`_muon.py_docs.md_docs.md`](./_muon.py_docs.md_docs.md)
- [`radam.py_kw.md_docs.md`](./radam.py_kw.md_docs.md)
- [`adamw.py_kw.md_docs.md`](./adamw.py_kw.md_docs.md)
- [`adagrad.py_kw.md_docs.md`](./adagrad.py_kw.md_docs.md)
- [`adadelta.py_docs.md_docs.md`](./adadelta.py_docs.md_docs.md)
- [`lbfgs.py_docs.md_docs.md`](./lbfgs.py_docs.md_docs.md)
- [`rmsprop.py_kw.md_docs.md`](./rmsprop.py_kw.md_docs.md)
- [`lbfgs.py_kw.md_docs.md`](./lbfgs.py_kw.md_docs.md)
- [`adamw.py_docs.md_docs.md`](./adamw.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `lr_scheduler.py_docs.md_docs.md`
- **Keyword Index**: `lr_scheduler.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
