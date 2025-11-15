# Documentation: `docs/torch/nn/modules/module.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/modules/module.py_docs.md`
- **Size**: 53,770 bytes (52.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/modules/module.py`

## File Metadata

- **Path**: `torch/nn/modules/module.py`
- **Size**: 127,526 bytes (124.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import functools
import inspect
import itertools
import warnings
import weakref
from collections import namedtuple, OrderedDict
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Optional, overload, TypeVar, Union
from typing_extensions import Self

import torch
from torch import device, dtype, Tensor
from torch._prims_common import DeviceLikeType
from torch.nn.parameter import Buffer, Parameter
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.hooks import BackwardHook, RemovableHandle


__all__ = [
    "register_module_forward_pre_hook",
    "register_module_forward_hook",
    "register_module_full_backward_pre_hook",
    "register_module_backward_hook",
    "register_module_full_backward_hook",
    "register_module_buffer_registration_hook",
    "register_module_module_registration_hook",
    "register_module_parameter_registration_hook",
    "Module",
]

_grad_t = Union[tuple[Tensor, ...], Tensor]
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar("T", bound="Module")


class _IncompatibleKeys(
    # pyrefly: ignore [invalid-inheritance]
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
):
    __slots__ = ()

    def __repr__(self) -> str:
        # pyrefly: ignore [missing-attribute]
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


r"""This tracks hooks common to all modules that are executed immediately before
.registering the buffer/module/parameter"""
_global_buffer_registration_hooks: dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: dict[int, Callable] = OrderedDict()


class _WrappedHook:
    def __init__(self, hook: Callable, module: Optional["Module"] = None) -> None:
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)

        self.with_module: bool = False

        if module is not None:
            self.module: weakref.ReferenceType[Module] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError("You are trying to call the hook of a dead Module!")
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> dict:
        result = {"hook": self.hook, "with_module": self.with_module}
        if self.with_module:
            # pyrefly: ignore [unsupported-operation]
            result["module"] = self.module()

        return result

    def __setstate__(self, state: dict):
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        if self.with_module:
            if state["module"] is None:
                raise RuntimeError(
                    "You are trying to revive the hook of a dead Module!"
                )
            self.module = weakref.ref(state["module"])


r"""This tracks hooks common to all modules that are executed before/after
calling forward and backward. This is global state used for debugging/profiling
purposes"""
_global_backward_pre_hooks: dict[int, Callable] = OrderedDict()
_global_backward_hooks: dict[int, Callable] = OrderedDict()
_global_is_full_backward_hook: Optional[bool] = None
_global_forward_pre_hooks: dict[int, Callable] = OrderedDict()
_global_forward_hooks: dict[int, Callable] = OrderedDict()
_global_forward_hooks_always_called: dict[int, bool] = OrderedDict()
_global_forward_hooks_with_kwargs: dict[int, bool] = OrderedDict()


def _has_any_global_hook():
    return (
        _global_backward_pre_hooks
        or _global_backward_hooks
        or _global_forward_pre_hooks
        or _global_forward_hooks
        or _global_forward_hooks_always_called
        or _global_forward_hooks_with_kwargs
    )


_EXTRA_STATE_KEY_SUFFIX = "_extra_state"


def register_module_buffer_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a buffer registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_buffer` is invoked.
    It should have the following signature::

        hook(module, name, buffer) -> None or new buffer

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_buffer_registration_hooks)
    _global_buffer_registration_hooks[handle.id] = hook
    return handle


def register_module_module_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a module registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_module` is invoked.
    It should have the following signature::

        hook(module, name, submodule) -> None or new submodule

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_module_registration_hooks)
    _global_module_registration_hooks[handle.id] = hook
    return handle


def register_module_parameter_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a parameter registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_parameter` is invoked.
    It should have the following signature::

        hook(module, name, param) -> None or new parameter

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_parameter_registration_hooks)
    _global_parameter_registration_hooks[handle.id] = hook
    return handle


def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a forward pre-hook common to all modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None or modified input

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the input. User can either return a tuple or a
    single modified value in the hook. We will wrap the value into a tuple
    if a single value is returned(unless that value is already a tuple).

    This hook has precedence over the specific module hooks registered with
    ``register_forward_pre_hook``.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(
    hook: Callable[..., None],
    *,
    with_kwargs: bool = False,
    always_call: bool = False,
) -> RemovableHandle:
    r"""Register a global forward hook for all the modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    You can optionally modify the output of the module by returning a new value
    that will replace the output from the :func:`forward` function.

    Parameters:
        hook (Callable): The user defined hook to be registered.
        always_call (bool): If ``True`` the ``hook`` will be run regardless of
            whether an exception is raised while calling the Module.
            Default: ``False``
    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
    handle = RemovableHandle(
        _global_forward_hooks, extra_dict=_global_forward_hooks_always_called
    )
    _global_forward_hooks[handle.id] = hook
    if with_kwargs:
        _global_forward_hooks_with_kwargs[handle.id] = True
    if always_call:
        _global_forward_hooks_always_called[handle.id] = True
    return handle


def register_module_backward_hook(
    hook: Callable[["Module", _grad_t, _grad_t], None | _grad_t],
) -> RemovableHandle:
    r"""Register a backward hook common to all the modules.

    This function is deprecated in favor of
    :func:`torch.nn.modules.module.register_module_full_backward_hook`
    and the behavior of this function will change in future versions.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is True:
        raise RuntimeError(
            "Cannot use both regular backward hooks and full backward hooks as a "
            "global Module hook. Please use only one of them."
        )

    _global_is_full_backward_hook = False

    handle = RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


def register_module_full_backward_pre_hook(
    hook: Callable[["Module", _grad_t], None | _grad_t],
) -> RemovableHandle:
    r"""Register a backward pre-hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    Hooks registered using this function behave in the same way as those
    registered by :meth:`torch.nn.Module.register_full_backward_pre_hook`.
    Refer to its documentation for more details.

    Hooks registered using this function will be called before hooks registered
    using :meth:`torch.nn.Module.register_full_backward_pre_hook`.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    handle = RemovableHandle(_global_backward_pre_hooks)
    _global_backward_pre_hooks[handle.id] = hook
    return handle


def register_module_full_backward_hook(
    hook: Callable[["Module", _grad_t, _grad_t], None | _grad_t],
) -> RemovableHandle:
    r"""Register a backward hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    Hooks registered using this function behave in the same way as those
    registered by :meth:`torch.nn.Module.register_full_backward_hook`.
    Refer to its documentation for more details.

    Hooks registered using this function will be called before hooks registered
    using :meth:`torch.nn.Module.register_full_backward_hook`.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is False:
        raise RuntimeError(
            "Cannot use both regular backward hooks and full backward hooks as a "
            "global Module hook. Please use only one of them."
        )

    _global_is_full_backward_hook = True

    handle = RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


# Trick mypy into not applying contravariance rules to inputs by defining
# forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
def _forward_unimplemented(self, *input: Any) -> None:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(
        f'Module [{type(self).__name__}] is missing the required "forward" function'
    )


class Module:
    r"""Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F


        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.

    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.

    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    """

    dump_patches: bool = False

    _version: int = 1
    r"""This allows better BC support for :meth:`load_state_dict`. In
    :meth:`state_dict`, the version number will be saved as in the attribute
    `_metadata` of the returned state dict, and thus pickled. `_metadata` is a
    dictionary with keys that follow the naming convention of state dict. See
    ``_load_from_state_dict`` on how to use this information in loading.

    If new parameters/buffers are added/removed from a module, this number shall
    be bumped, and the module's `_load_from_state_dict` method can compare the
    version number and do appropriate changes if the state dict is from before
    the change."""

    training: bool
    _parameters: dict[str, Optional[Parameter]]
    _buffers: dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: set[str]
    _backward_pre_hooks: dict[int, Callable]
    _backward_hooks: dict[int, Callable]
    _is_full_backward_hook: Optional[bool]
    _forward_hooks: dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_hooks_with_kwargs: dict[int, bool]
    # forward hooks that should always be called even if an exception is raised
    _forward_hooks_always_called: dict[int, bool]
    _forward_pre_hooks: dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_pre_hooks_with_kwargs: dict[int, bool]
    _state_dict_hooks: dict[int, Callable]
    _load_state_dict_pre_hooks: dict[int, Callable]
    _state_dict_pre_hooks: dict[int, Callable]
    _load_state_dict_post_hooks: dict[int, Callable]
    _modules: dict[str, Optional["Module"]]
    call_super_init: bool = False
    _compiled_call_impl: Optional[Callable] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize internal Module state, shared by both nn.Module and ScriptModule."""
        torch._C._log_api_usage_once("python.nn_module")

        # Backward compatibility: no args used to be allowed when call_super_init=False
        if self.call_super_init is False and bool(kwargs):
            raise TypeError(
                f"{type(self).__name__}.__init__() got an unexpected keyword argument '{next(iter(kwargs))}'"
                ""
            )

        if self.call_super_init is False and bool(args):
            raise TypeError(
                f"{type(self).__name__}.__init__() takes 1 positional argument but {len(args) + 1} were"
                " given"
            )

        """
        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Module.__setattr__ overhead. Module's __setattr__ has special
        handling for parameters, submodules, and buffers but simply calls into
        super().__setattr__ for all other attributes.
        """
        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        super().__setattr__("_buffers", {})
        super().__setattr__("_non_persistent_buffers_set", set())
        super().__setattr__("_backward_pre_hooks", OrderedDict())
        super().__setattr__("_backward_hooks", OrderedDict())
        super().__setattr__("_is_full_backward_hook", None)
        super().__setattr__("_forward_hooks", OrderedDict())
        super().__setattr__("_forward_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_forward_hooks_always_called", OrderedDict())
        super().__setattr__("_forward_pre_hooks", OrderedDict())
        super().__setattr__("_forward_pre_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_state_dict_hooks", OrderedDict())
        super().__setattr__("_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_post_hooks", OrderedDict())
        super().__setattr__("_modules", {})

        if self.call_super_init:
            super().__init__(*args, **kwargs)

    forward: Callable[..., Any] = _forward_unimplemented

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule does not support non-persistent buffers")

        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(
                f"buffer name should be a string. Got {torch.typename(name)}"
            )
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif tensor is not None and not (
            isinstance(tensor, torch.Tensor) or hasattr(tensor, "__torch_function__")
        ):
            raise TypeError(
                f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
                "(torch Tensor or None required)"
            )
        else:
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                if output is not None:
                    tensor = output
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        elif not isinstance(name, str):
            raise TypeError(
                f"parameter name should be a string. Got {torch.typename(name)}"
            )
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
                "(torch.nn.Parameter or None required)"
            )
        elif param.grad_fn:
            raise ValueError(
                f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
                f"parameters must be created explicitly. To express '{name}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method."
            )
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{torch.typename(module)} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(
                f"module name should be a string. Got {torch.typename(name)}"
            )
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError(f'module name can\'t contain ".", got: {name}')
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            if output is not None:
                module = output
        self._modules[name] = module

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If at any point along the path resulting from
                the target string the (sub)path resolves to a non-existent
                attribute name or an object that is not an instance of ``nn.Module``.
        """
        if target == "":
            return self

        atoms: list[str] = target.split(".")
        mod: torch.nn.Module = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")

        return mod

    def set_submodule(
        self, target: str, module: "Module", strict: bool = False
    ) -> None:
        """
        Set the submodule given by ``target`` if it exists, otherwise throw an error.

        .. note::
            If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
            or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
            the method will only attempt to replace an existing submodule and throw an error if
            the submodule does not exist.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(3, 3, 3)
                    )
                    (linear): Linear(3, 3)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To override the ``Conv2d`` with a new submodule ``Linear``, you
        could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
        where ``strict`` could be ``True`` or ``False``

        To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
        you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

        In the above if you set ``strict=True`` and call
        ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
        will be raised because ``net_b`` does not have a submodule named ``conv``.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)
            module: The module to set the submodule to.
            strict: If ``False``, the method will replace an existing submodule
                or create a new submodule if the parent module exists. If ``True``,
                the method will only attempt to replace an existing submodule and throw an error
                if the submodule doesn't already exist.

        Raises:
            ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
            AttributeError: If at any point along the path resulting from
                the ``target`` string the (sub)path resolves to a non-existent
                attribute name or an object that is not an instance of ``nn.Module``.
        """
        if target == "":
            raise ValueError("Cannot set the submodule without a target name!")

        atoms: list[str] = target.split(".")
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                "`" + "module" + f"` is not an nn.Module, found {type(module)}"
            )
        if len(atoms) == 1:
            parent: torch.nn.Module = self
        else:
            parent_key = ".".join(atoms[:-1])
            parent = self.get_submodule(parent_key)

        if strict and not hasattr(parent, atoms[-1]):
            raise AttributeError(
                parent._get_name() + " has no attribute `" + atoms[-1] + "`"
            )
        if hasattr(parent, atoms[-1]):
            mod = getattr(parent, atoms[-1])
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + atoms[-1] + "` is not an nn.Module")
        setattr(parent, atoms[-1], module)

    def get_parameter(self, target: str) -> "Parameter":
        """Return the parameter given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        module_path, _, param_name = target.rpartition(".")

        mod: torch.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + param_name + "`"
            )

        param: torch.nn.Parameter = getattr(mod, param_name)

        if not isinstance(param, torch.nn.Parameter):
            raise AttributeError("`" + param_name + "` is not an nn.Parameter")

        return param

    def get_buffer(self, target: str) -> "Tensor":
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod: torch.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + buffer_name + "`"
            )

        buffer: torch.Tensor = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    def get_extra_state(self) -> Any:
        """Return any extra state to include in the module's state_dict.

        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict. We only provide backwards compatibility guarantees
        for serializing Tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object: Any extra state to store in the module's state_dict
        """
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )

    def set_extra_state(self, state: Any) -> None:
        """Set extra state contained in the loaded `state_dict`.

        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`
        """
        raise RuntimeError(
            "Reached a code path in Module.set_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )

    def _apply(self, fn, recurse=True):
        if recurse:
            for module in self.children():
                module._apply(fn)

        from torch._subclasses.fake_tensor import FakeTensor

        def compute_should_use_set_data(tensor, tensor_applied) -> bool:
            if torch._has_compatible_shallow_copy_type(
                tensor, tensor_applied
            ) and not isinstance(tensor_applied, FakeTensor):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        should_use_swap_tensors = (
            torch.__future__.get_swap_module_params_on_conversion()
        )

        for key, param in self._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                param_applied = fn(param)
            p_should_use_set_data = compute_should_use_set_data(param, param_applied)

            # subclasses may have multiple child tensors so we need to use swap_tensors
            p_should_use_swap_tensors = (
                should_use_swap_tensors
                or is_traceable_wrapper_subclass(param_applied)
                or isinstance(param, FakeTensor)
            )

            param_grad = param.grad
            if p_should_use_swap_tensors:
                try:
                    if param_grad is not None:
                        # Accessing param.grad makes its at::Tensor's use_count 2, which will prevent swapping.
                        # Decrement use count of the gradient by setting to None
                        param.grad = None
                    param_applied = torch.nn.Parameter(
                        # pyrefly: ignore [bad-argument-type]
                        param_applied,
                        requires_grad=param.requires_grad,
                    )
                    torch.utils.swap_tensors(param, param_applied)
                except Exception as e:
                    if param_grad is not None:
                        param.grad = param_grad
                    raise RuntimeError(
                        f"_apply(): Couldn't swap {self._get_name()}.{key}"
                    ) from e
                out_param = param
            elif p_should_use_set_data:
                # pyrefly: ignore [bad-assignment]
                param.data = param_applied
                out_param = param
            else:
                assert isinstance(param, Parameter)
                assert param.is_leaf
                # pyrefly: ignore [bad-argument-type]
                out_param = Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            if param_grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param_grad)
                g_should_use_set_data = compute_should_use_set_data(
                    param_grad, grad_applied
                )
                if p_should_use_swap_tensors:
                    grad_applied.requires_grad_(param_grad.requires_grad)
                    try:
                        torch.utils.swap_tensors(param_grad, grad_applied)
                    except Exception as e:
                        raise RuntimeError(
                            f"_apply(): Couldn't swap {self._get_name()}.{key}.grad"
                        ) from e
                    out_param.grad = param_grad
                elif g_should_use_set_data:
                    assert out_param.grad is not None
                    out_param.grad.data = grad_applied
                else:
                    assert param_grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(
                        param_grad.requires_grad
                    )

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self, fn: Callable[["Module"], None]) -> Self:
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) is nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )

        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def cuda(self, device: Optional[int | device] = None) -> Self:
        r"""Move all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))

    def ipu(self, device: Optional[int | device] = None) -> Self:
        r"""Move all model parameters and buffers to the IPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on IPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.ipu(device))

    def xpu(self, device: Optional[int | device] = None) -> Self:
        r"""Move all model parameters and buffers to the XPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on XPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.xpu(device))

    def mtia(self, device: Optional[int | device] = None) -> Self:
        r"""Move all model parameters and buffers to the MTIA.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on MTIA while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.mtia(device))

    def cpu(self) -> Self:
        r"""Move all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())

    def type(self, dst_type: dtype | str) -> Self:
        r"""Casts all parameters and buffers to :attr:`dst_type`.

        .. note::
            This method modifies the module in-place.

        Args:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.type(dst_type))

    def float(self) -> Self:
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self) -> Self:
        r"""Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def half(self) -> Self:
        r"""Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    def bfloat16(self) -> Self:
        r"""Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)

    def to_empty(
        self, *, device: Optional[DeviceLikeType], recurse: bool = True
    ) -> Self:
        r"""Move the parameters and buffers to the specified device without copying storage.

        Args:
            device (:class:`torch.device`): The desired device of the parameters
                and buffers in this module.
            recurse (bool): Whether parameters and buffers of submodules should
                be recursively moved to the specified device.

        Returns:
            Module: self
        """
        return self._apply(
            lambda t: torch.empty_like(t, device=device), recurse=recurse
        )

    @overload
    def to(
        self,
        device: Optional[DeviceLikeType] = ...,
        dtype: Optional[dtype] = ...,
        non_blocking: bool = ...,
    ) -> Self: ...

    @overload
    def to(self, dtype: dtype, non_blocking: bool = ...) -> Self: ...

    @overload
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> Self: ...

    def to(self, *args, **kwargs):
        r"""Move and/or cast the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/nn/modules`):

- [`sparse.py_docs.md_docs.md`](./sparse.py_docs.md_docs.md)
- [`instancenorm.py_kw.md_docs.md`](./instancenorm.py_kw.md_docs.md)
- [`activation.py_kw.md_docs.md`](./activation.py_kw.md_docs.md)
- [`container.py_docs.md_docs.md`](./container.py_docs.md_docs.md)
- [`distance.py_kw.md_docs.md`](./distance.py_kw.md_docs.md)
- [`pixelshuffle.py_kw.md_docs.md`](./pixelshuffle.py_kw.md_docs.md)
- [`batchnorm.py_docs.md_docs.md`](./batchnorm.py_docs.md_docs.md)
- [`transformer.py_kw.md_docs.md`](./transformer.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `module.py_docs.md_docs.md`
- **Keyword Index**: `module.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
