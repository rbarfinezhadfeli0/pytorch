# Documentation: `torch/_tensor.py`

## File Metadata

- **Path**: `torch/_tensor.py`
- **Size**: 76,306 bytes (74.52 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copyreg
import enum
import functools
import itertools
import warnings
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from numbers import Number
from typing import Any, cast, Concatenate, Optional, TypeVar, Union
from typing_extensions import ParamSpec

import torch
import torch._C as _C
from torch._namedtensor_internals import (
    check_serializing_named_tensor,
    is_ellipsis,
    resolve_ellipsis,
    single_ellipsis_index,
    unzip_namedshape,
    update_names,
)
from torch.overrides import (
    get_default_nowrap_functions,
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)


_P = ParamSpec("_P")
_TensorLike = TypeVar("_TensorLike", bound=_C.TensorBase)


def _handle_torch_function_and_wrap_type_error_to_not_implemented(
    f: Callable[Concatenate[_TensorLike, _P], "Tensor"],
) -> Callable[Concatenate[_TensorLike, _P], "Tensor"]:
    @functools.wraps(f)
    def wrapped(self: _TensorLike, *args: _P.args, **kwargs: _P.kwargs) -> "Tensor":
        try:
            # See https://github.com/pytorch/pytorch/issues/75462
            sargs = self, *args
            if has_torch_function(sargs):
                return handle_torch_function(wrapped, sargs, *sargs, **kwargs)
            return f(self, *args, **kwargs)
        except TypeError:
            return NotImplemented

    return wrapped


# Should not be used, this is kept only for BC of loading old serialized Tensor subclasses
def _rebuild_from_type(func, type, args, dict):
    if type is Tensor:
        return func(*args)

    ret = func(*args).as_subclass(type)
    ret.__dict__ = dict
    return ret


def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    if type(ret) is not new_type:
        ret = ret.as_subclass(new_type)
    # Tensor does define __setstate__ even though it doesn't define
    # __getstate__. So only use __setstate__ if it is NOT the one defined
    # on Tensor
    if (
        getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
        is not Tensor.__setstate__
    ):
        ret.__setstate__(state)
    else:
        ret = torch._utils._set_obj_state(ret, state)
    return ret


def _dtype_to_typestr(dtype):
    # CUDA devices are little-endian and tensors are stored in native byte
    # order. 1-byte entries are endian-agnostic.
    return {
        torch.complex64: "<c8",
        torch.complex128: "<c16",
        torch.bfloat16: "<V2",  # Same as ml_dtypes.bfloat16.dtype.str.
        torch.float16: "<f2",
        torch.float32: "<f4",
        torch.float64: "<f8",
        torch.uint8: "|u1",
        torch.int8: "|i1",
        torch.uint16: "<u2",
        torch.int16: "<i2",
        torch.uint32: "<u4",
        torch.int32: "<i4",
        torch.uint64: "<u8",
        torch.int64: "<i8",
        torch.bool: "|b1",
    }[dtype]


# NB: If you subclass Tensor, and want to share the subclassed class
# across processes, you must also update torch/multiprocessing/reductions.py
# to define a ForkingPickler serialization mode for the class.
#
# NB: If you add a new method to Tensor, you must update
# torch/_C/__init__.pyi.in to add a type annotation for your method;
# otherwise, it will not show up in autocomplete.
class Tensor(torch._C.TensorBase):
    _is_param: bool

    def _clear_non_serializable_cached_data(self):
        r"""Clears any data cached in the tensor's ``__dict__`` that would prevent the tensor
        from being serialized.

        For example, subclasses with custom dispatched sizes / strides cache this info in
        non-serializable PyCapsules within the ``__dict__``, and this must be cleared out for
        serialization to function.

        Any subclass that overrides this MUST call ``super()._clear_non_serializable_cached_data().``
        Additional data cleared within the override must be able to be re-cached transparently
        to avoid breaking subclass functionality.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor._clear_non_serializable_cached_data, (self,), self
            )
        # NB: Wrapper subclasses that implement custom-dispatched sizes / strides cache
        # this info via non-serializable PyCapsules.
        CACHED_SIZES_STRIDES_KEYS = [
            "_sym_sizes_capsule",
            "_sym_sizes_capsule_len",
            "_sym_strides_capsule",
            "_sym_strides_capsule_len",
        ]
        for key in CACHED_SIZES_STRIDES_KEYS:
            self.__dict__.pop(key, None)

    def __deepcopy__(self, memo):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__deepcopy__, (self,), self, memo)
        if not self.is_leaf:
            raise RuntimeError(
                "Only Tensors created explicitly by the user "
                "(graph leaves) support the deepcopy protocol at the moment.  "
                "If you were attempting to deepcopy a module, this may be because "
                "of a torch.nn.utils.weight_norm usage, "
                "see https://github.com/pytorch/pytorch/pull/103001"
            )
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            # TODO: skipping storage copy is wrong for meta, as meta
            # does accurate alias tracking; however, the code below
            # doesn't work because of
            # https://github.com/pytorch/pytorch/issues/47442
            # Update the test in test_serialization if you remove 'meta' from here
            if (
                self.is_sparse
                or self.device.type
                in ["lazy", "xla", "mtia", "mps", "maia", "meta", "ipu"]
                or (
                    not torch._C._has_storage(self)
                    and self.device.type == torch._C._get_privateuse1_backend_name()
                )
                or (type(self) is not Tensor and self.data_ptr() == 0)
            ):
                new_tensor = self.clone()
                if type(new_tensor) is not type(self):
                    raise RuntimeError(
                        "The default implementation of __deepcopy__() for wrapper subclasses "
                        "only works for subclass types that implement clone() and for which "
                        "cloning returns another instance of the same subclass. You should either "
                        "properly implement clone() for your subclass or override __deepcopy__() "
                        "if it is intended behavior for clone() to return an instance of a "
                        "different type."
                    )
            else:
                new_storage = self._typed_storage()._deepcopy(memo)
                if self.is_quantized:
                    # quantizer_params can be different type based on torch attribute
                    quantizer_params: Union[
                        tuple[torch.qscheme, float, int],
                        tuple[torch.qscheme, Tensor, Tensor, int],
                    ]
                    if self.qscheme() == torch.per_tensor_affine:
                        quantizer_params = (
                            self.qscheme(),
                            self.q_scale(),
                            self.q_zero_point(),
                        )
                    elif self.qscheme() in (
                        torch.per_channel_affine,
                        torch.per_channel_affine_float_qparams,
                    ):
                        quantizer_params = (
                            self.qscheme(),
                            self.q_per_channel_scales(),
                            self.q_per_channel_zero_points(),
                            self.q_per_channel_axis(),
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported qscheme {self.qscheme()} in deepcopy"
                        )
                    # TODO: Once we decide to break serialization FC, no longer
                    # need to wrap with TypedStorage
                    new_tensor = torch._utils._rebuild_qtensor(
                        torch.storage.TypedStorage(
                            wrap_storage=new_storage._untyped_storage,
                            dtype=self.dtype,
                            _internal=True,
                        ),
                        self.storage_offset(),
                        self.size(),
                        self.stride(),
                        quantizer_params,
                        self.requires_grad,
                        self._backward_hooks,
                    )
                    if type(new_tensor) is not type(self):
                        raise RuntimeError(
                            "The default implementation of __deepcopy__() for quantized tensors "
                            "expects the tensor returned by torch._utils._rebuild_qtensor() to "
                            "match the type of the instance being copied. If you encounter this, "
                            "please open an issue on PyTorch's GitHub."
                        )
                else:
                    new_tensor = self.new_empty([])
                    if type(new_tensor) is not type(self):
                        raise RuntimeError(
                            "The default implementation of __deepcopy__() for non-wrapper subclasses "
                            "only works for subclass types that implement new_empty() and for which "
                            "that function returns another instance of the same subclass. You should "
                            "either properly implement new_empty() for your subclass or override "
                            "__deepcopy__() if it is intended behavior for new_empty() to return "
                            "an instance of a different type."
                        )
                    new_tensor.set_(
                        new_storage, self.storage_offset(), self.size(), self.stride()
                    )
                    if self.is_conj():
                        new_tensor = new_tensor.conj_physical()
                    if self.is_neg():
                        new_tensor = new_tensor.neg()
            if self.requires_grad:
                new_tensor.requires_grad_()
            if self.grad is not None:
                new_tensor.grad = self.grad.__deepcopy__(memo)

            if type(self) is not Tensor:
                if type(new_tensor) is not type(self):
                    raise RuntimeError(
                        "Type of deepcopy result does not match the type of the source tensor. "
                        "If you encounter this, please open an issue on PyTorch's GitHub."
                    )

                # Plain Tensors don't have slots
                slots_to_save = copyreg._slotnames(self.__class__)  # type: ignore[attr-defined]
                for slot in slots_to_save:
                    if hasattr(self, slot):
                        setattr(new_tensor, slot, deepcopy(getattr(self, slot), memo))

            # don't try to deepcopy non-serializable cached data
            self._clear_non_serializable_cached_data()
            new_tensor.__dict__ = deepcopy(self.__dict__, memo)

            memo[id(self)] = new_tensor
            return new_tensor

    def __reduce_ex__(self, proto):
        materialize_fake_tensors = (
            torch.serialization._serialization_tls.materialize_fake_tensors
        )
        state = torch._utils._get_obj_state(self)
        # Ignore all state when using FakeTensor with skip_data(materialize_fake_tensors) because FakeTensor has
        # some state that cannot be pickled
        if (
            # TODO: remove hasattr, it's a hack to support versions of torch that
            # don't have _subclasses
            hasattr(torch, "_subclasses")
            and type(self) is torch._subclasses.fake_tensor.FakeTensor
            and materialize_fake_tensors
        ) or (type(self) is Tensor and not state):
            # Fast path for regular tensor without Python state.
            return self._reduce_ex_internal(proto)
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reduce_ex__, (self,), self, proto)
        func, args = self._reduce_ex_internal(proto)
        # sizes / strides cache needs to be cleared here because it'll just be re-cached
        # if cleared earlier. Note that state references the -actual- tensor dict.
        self._clear_non_serializable_cached_data()
        return (_rebuild_from_type_v2, (func, type(self), args, state))

    def storage(self):
        r"""
        storage() -> torch.TypedStorage

        Returns the underlying :class:`TypedStorage`.

        .. warning::

            :class:`TypedStorage` is deprecated. It will be removed in the future, and
            :class:`UntypedStorage` will be the only storage class. To access the
            :class:`UntypedStorage` directly, use :attr:`Tensor.untyped_storage()`.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.storage, (self,), self)

        torch.storage._warn_typed_storage_removal(stacklevel=2)
        return self._typed_storage()

    # For internal use only, to avoid raising deprecation warning
    def _typed_storage(self):
        untyped_storage = self.untyped_storage()
        return torch.TypedStorage(
            wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
        )

    def _reduce_ex_internal(self, proto):
        check_serializing_named_tensor(self)

        from torch.utils.hooks import warn_if_has_hooks

        # See Note [Don't serialize hooks]
        warn_if_has_hooks(self)
        backward_hooks: dict[Any, Any] = OrderedDict()

        skip_data = torch.serialization._serialization_tls.skip_data
        materialize_fake_tensors = (
            torch.serialization._serialization_tls.materialize_fake_tensors
        )

        if self.device.type in ["xla", "maia", "mtia"] or (
            not torch._C._has_storage(self)
            and self.device.type == torch._C._get_privateuse1_backend_name()
        ):
            if skip_data:
                raise RuntimeError(
                    "Cannot serialize tensors on backends with no storage under skip_data context manager"
                )
            cpu_tensor = self.cpu()
            return (
                torch._utils._rebuild_device_tensor_from_cpu_tensor,
                (cpu_tensor, self.dtype, str(self.device), self.requires_grad),
            )
        if self.device.type == "meta":
            # NB: This implementation BREAKS storage sharing.  Current
            # hypothesis is that no one cares for meta tensors.
            if skip_data:
                warnings.warn(
                    "Serializing tensors on the meta device under skip_data context manager is a no-op",
                    stacklevel=2,
                )
            arg_meta = (
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.requires_grad,
            )
            return (torch._utils._rebuild_meta_tensor_no_storage, arg_meta)
        if self.is_quantized:
            if skip_data:
                raise RuntimeError(
                    "Cannot serialize qtensor under skip_data context manager, file an issue if you need this feature"
                )
            # quantizer_params can be different type based on torch attribute
            quantizer_params: Union[
                tuple[torch.qscheme, float, int], tuple[Any, Tensor, Tensor, int]
            ]
            if self.qscheme() == torch.per_tensor_affine:
                quantizer_params = (
                    torch.per_tensor_affine,
                    self.q_scale(),
                    self.q_zero_point(),
                )
            elif self.qscheme() in (
                torch.per_channel_affine,
                torch.per_channel_affine_float_qparams,
            ):
                # convert scales and zero points to tuple to avoid recursive calls
                # when/if we get multi-axis quantized tensors in the future, the shape
                # is recoverable from the main tensor shape
                quantizer_params = (
                    torch.per_channel_affine,
                    self.q_per_channel_scales(),
                    self.q_per_channel_zero_points(),
                    self.q_per_channel_axis(),
                )
            else:
                raise RuntimeError(
                    f"Serialization is not supported for tensors of type {self.qscheme()}"
                )
            # TODO: Once we decide to break serialization FC, no longer
            # need to wrap with TypedStorage
            args_qtensor = (
                torch.storage.TypedStorage(
                    wrap_storage=self._typed_storage()._untyped_storage,
                    dtype=self.dtype,
                    _internal=True,
                ),
                self.storage_offset(),
                tuple(self.size()),
                self.stride(),
                quantizer_params,
                self.requires_grad,
                backward_hooks,
            )
            return (torch._utils._rebuild_qtensor, args_qtensor)
        elif self.is_sparse:
            if self.layout == torch.sparse_coo:
                args_sparse = (
                    self.layout,
                    (self._indices(), self._values(), self.size(), self.is_coalesced()),
                )
            else:
                raise NotImplementedError(
                    f"sparse tensor __reduce_ex__ for layout `{self.layout}`"
                )
            return (torch._utils._rebuild_sparse_tensor, args_sparse)
        elif self.layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices, plain_indices = (
                    self.crow_indices(),
                    self.col_indices(),
                )
            else:
                compressed_indices, plain_indices = (
                    self.ccol_indices(),
                    self.row_indices(),
                )
            args_sparse_compressed = (
                self.layout,
                (
                    compressed_indices,
                    plain_indices,
                    self.values(),
                    self.size(),
                ),
            )
            return (torch._utils._rebuild_sparse_tensor, args_sparse_compressed)
        elif self.is_nested:
            if skip_data:
                raise RuntimeError(
                    "Cannot serialize nested tensor under skip_data context manager, file an issue if you need this feature"
                )
            args_nested = (
                # NB: values() currently returns the storage as a buffer in an unsafe way.
                # Ideally, we'd use a private API for this instead. TODO: Switch to this if
                # we ever get around to adding it.
                self.values(),
                self._nested_tensor_size(),
                self._nested_tensor_strides(),
                self._nested_tensor_storage_offsets(),
            )
            return (torch._utils._rebuild_nested_tensor, args_nested)
        elif (
            type(self) is not torch.Tensor
            and type(self).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__
            and (
                isinstance(self, torch._subclasses.functional_tensor.FunctionalTensor)
                or (
                    not isinstance(self, torch._subclasses.fake_tensor.FakeTensor)
                    and self.data_ptr() == 0
                )
            )
        ):
            arg_wrapper_subclass = (
                type(self),
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.storage_offset(),
                self.layout,
                self.device,
                self.requires_grad,
            )
            return (torch._utils._rebuild_wrapper_subclass, arg_wrapper_subclass)
        elif (
            type(self) is not torch.Tensor
            and type(self).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__
            and (
                isinstance(self, torch._subclasses.fake_tensor.FakeTensor)
                and not (skip_data and materialize_fake_tensors)
            )
        ):
            arg_wrapper_subclass = (
                type(self),
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.storage_offset(),
                self.layout,
                self.device,
                self.requires_grad,
            )
            return (torch._utils._rebuild_wrapper_subclass, arg_wrapper_subclass)
        else:
            v3_dtypes = torch.storage._new_dtypes()
            if self.dtype in v3_dtypes:
                rebuild_func = torch._utils._rebuild_tensor_v3
                storage = self.untyped_storage()
            else:
                # TODO: Once we decide to break serialization FC, no longer
                # need to wrap with TypedStorage
                rebuild_func = torch._utils._rebuild_tensor_v2  # type: ignore[assignment]
                storage = torch.storage.TypedStorage(
                    wrap_storage=self._typed_storage()._untyped_storage,
                    dtype=self.dtype,
                    _internal=True,
                )  # type: ignore[assignment]

            # TODO: remove hasattr, it's a hack to support versions of torch that
            # don't have _subclasses
            if (
                hasattr(torch, "_subclasses")
                and isinstance(self, torch._subclasses.fake_tensor.FakeTensor)
                and skip_data
            ):
                storage._fake_device = self.device

            args = (
                storage,
                self.storage_offset(),
                tuple(self.size()),
                self.stride(),
                self.requires_grad,
                backward_hooks,
            )  # previously was self._backward_hooks

            if isinstance(storage, torch.storage.UntypedStorage):
                args = args + (self.dtype,)  # type: ignore[assignment]

            metadata = torch._utils.get_tensor_metadata(self)
            if metadata:
                args = args + (metadata,)  # type: ignore[assignment]

            return (rebuild_func, args)

    def __setstate__(self, state):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__setstate__, (self,), self, state)
        # Warning: this method is NOT called when you torch.load() a tensor;
        # that is managed by _rebuild_tensor_v2
        if not self.is_leaf:
            raise RuntimeError("__setstate__ can be only called on leaf Tensors")
        if len(state) == 4:
            # legacy serialization of Tensor
            # pyrefly: ignore [not-iterable]
            self.set_(*state)
            return
        elif len(state) == 5:
            # legacy serialization of Variable
            self.data = state[0]
            state = (state[3], state[4], state[2])
        # The setting of _backward_hooks is expected to be a no-op.
        # See Note [Don't serialize hooks]
        self.requires_grad, _, self._backward_hooks = state

    def __repr__(self, *, tensor_contents=None):
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.__repr__, (self,), self, tensor_contents=tensor_contents
            )
        # All strings are unicode in Python 3.
        return torch._tensor_str._str(self, tensor_contents=tensor_contents)

    def backward(
        self, gradient=None, retain_graph=None, create_graph=False, inputs=None
    ):
        r"""Computes the gradient of current tensor wrt graph leaves.

        The graph is differentiated using the chain rule. If the tensor is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionally requires specifying a ``gradient``.
        It should be a tensor of matching type and shape, that represents
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to zero
        ``.grad`` attributes or set them to ``None`` before calling it.
        See :ref:`Default gradient layouts<default-grad-layouts>`
        for details on the memory layout of accumulated gradients.

        .. note::

            If you run any forward ops, create ``gradient``, and/or call ``backward``
            in a user-specified CUDA stream context, see
            :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

        .. note::

            When ``inputs`` are provided and a given input is not a leaf,
            the current implementation will call its grad_fn (though it is not strictly needed to get this gradients).
            It is an implementation detail on which the user should not rely.
            See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

        Args:
            gradient (Tensor, optional): The gradient of the function
                being differentiated w.r.t. ``self``.
                This argument can be omitted if ``self`` is a scalar. Defaults to ``None``.
            retain_graph (bool, optional): If ``False``, the graph used to compute the grads will be freed;
                If ``True``, it will be retained. The default is ``None``, in which case the value is inferred from ``create_graph``
                (i.e., the graph is retained only when higher-order derivative tracking is requested). Note that in nearly all cases
                setting this option to True is not needed and often can be worked around in a much more efficient way.
            create_graph (bool, optional): If ``True``, graph of the derivative will
                be constructed, allowing to compute higher order derivative
                products. Defaults to ``False``.
            inputs (Sequence[Tensor], optional): Inputs w.r.t. which the gradient will be
                accumulated into ``.grad``. All other tensors will be ignored. If not
                provided, the gradient is accumulated into all the leaf Tensors that were
                used to compute the :attr:`tensors`. Defaults to ``None``.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.backward,
                (self,),
                self,
                gradient=gradient,
                retain_graph=retain_graph,
                create_graph=create_graph,
                inputs=inputs,
            )
        torch.autograd.backward(
            self, gradient, retain_graph, create_graph, inputs=inputs
        )

    def index(self, positions, dims):
        """
        Index a regular tensor by binding specified positions to dims.

        This converts a regular tensor to a first-class tensor by binding
        the specified positional dimensions to Dim objects.

        Args:
            positions: Tuple of dimension positions to bind
            dims: Dim objects or tuple of Dim objects to bind to

        Returns:
            First-class tensor with specified dimensions bound
        """
        # TODO: make it possible to dispatch on positions/dims
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.index,
                (self,),
                self,
                positions,
                dims,
            )

        from functorch.dim import index

        return index(self, positions, dims)

    def register_hook(self, hook):
        r"""Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        Tensor is computed. The hook should have the following signature::

            hook(grad) -> Tensor or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v.grad

             2
             4
             6
            [torch.FloatTensor of size (3,)]

            >>> h.remove()  # removes the hook
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.register_hook, (self,), self, hook)
        if not self.requires_grad:
            raise RuntimeError(
                "cannot register a hook on a tensor that doesn't require gradient"
            )
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None:
                self.grad_fn._register_hook_dict(self)

        from torch.utils.hooks import RemovableHandle

        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_post_accumulate_grad_hook(self, hook):
        r"""Registers a backward hook that runs after grad accumulation.

        The hook will be called after all gradients for a tensor have been accumulated,
        meaning that the .grad field has been updated on that tensor. The post
        accumulate grad hook is ONLY applicable for leaf tensors (tensors without a
        .grad_fn field). Registering this hook on a non-leaf tensor will error!

        The hook should have the following signature::

            hook(param: Tensor) -> None

        Note that, unlike other autograd hooks, this hook operates on the tensor
        that requires grad and not the grad itself. The hook can in-place modify
        and access its Tensor argument, including its .grad field.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks. Since
            this hook runs during the backward pass, it will run in no_grad mode (unless
            create_graph is True). You can use torch.enable_grad() to re-enable autograd
            within the hook if you need it.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> lr = 0.01
            >>> # simulate a simple SGD update
            >>> h = v.register_post_accumulate_grad_hook(lambda p: p.add_(p.grad, alpha=-lr))
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v
            tensor([-0.0100, -0.0200, -0.0300], requires_grad=True)

            >>> h.remove()  # removes the hook
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.register_post_accumulate_grad_hook, (self,), self, hook
            )
        if not self.requires_grad:
            raise RuntimeError(
                "cannot register a hook on a tensor that doesn't require gradient"
            )
        if self.grad_fn is not None:
            raise RuntimeError(
                "post accumulate grad hooks cannot be registered on non-leaf tensors"
            )
        if self._post_accumulate_grad_hooks is None:
            self._post_accumulate_grad_hooks: dict[Any, Any] = (
                # pyrefly: ignore [bad-assignment]
                OrderedDict()
            )

        from torch.utils.hooks import RemovableHandle

        handle = RemovableHandle(self._post_accumulate_grad_hooks)
        self._post_accumulate_grad_hooks[handle.id] = hook
        return handle

    def reinforce(self, reward):
        def trim(str):
            return "\n".join([line.strip() for line in str.split("\n")])

        raise RuntimeError(
            trim(
                r"""reinforce() was removed.
            Use torch.distributions instead.
            See https://pytorch.org/docs/main/distributions.html

            Instead of:

            probs = policy_network(state)
            action = probs.multinomial()
            next_state, reward = env.step(action)
            action.reinforce(reward)
            action.backward()

            Use:

            probs = policy_network(state)
            # NOTE: categorical is equivalent to what used to be called multinomial
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            next_state, reward = env.step(action)
            loss = -m.log_prob(action) * reward
            loss.backward()
        """
            )
        )

    detach = _C._add_docstr(
        _C.TensorBase.detach,
        r"""
    Returns a new Tensor, detached from the current graph.

    The result will never require gradient.

    This method also affects forward mode AD gradients and the result will never
    have forward mode AD gradients.

    .. note::

      Returned Tensor shares the same storage with the original one.
      In-place modifications on either of them will be seen, and may trigger
      errors in correctness checks.
    """,
    )

    detach_ = _C._add_docstr(
        _C.TensorBase.detach_,
        r"""
    Detaches the Tensor from the graph that created it, making it a leaf.
    Views cannot be detached in-place.

    This method also affects forward mode AD gradients and the result will never
    have forward mode AD gradients.
    """,
    )

    def is_shared(self):
        r"""Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.is_shared, (self,), self)
        return self._typed_storage()._is_shared()

    def share_memory_(self):
        r"""Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.

        See :meth:`torch.UntypedStorage.share_memory_` for more details.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.share_memory_, (self,), self)
        self._typed_storage()._share_memory_()
        return self

    def module_load(self, other, assign=False):
        r"""Defines how to transform ``other`` when loading it into ``self`` in :meth:`~nn.Module.load_state_dict`.

        Used when :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        It is expected that ``self`` is a parameter or buffer in an ``nn.Module`` and ``other`` is the
        value in the state dictionary with the corresponding key, this method defines
        how ``other`` is remapped before being swapped with ``self`` via
        :func:`~torch.utils.swap_tensors` in :meth:`~nn.Module.load_state_dict`.

        .. note::
            This method should always return a new object that is not ``self`` or ``other``.
            For example, the default implementation returns ``self.copy_(other).detach()``
            if ``assign`` is ``False`` or ``other.detach()`` if ``assign`` is ``True``.

        Args:
            other (Tensor): value in state dict with key corresponding to ``self``
            assign (bool): the assign argument passed to :meth:`nn.Module.load_state_dict`

        """
        if has_torch_function_variadic(self, other):
            return handle_torch_function(
                Tensor.module_load, (self, other), self, other, assign=assign
            )

        if assign:
            return other.detach()
        else:
            return self.copy_(other).detach()

    def __reversed__(self):
        r"""Reverses the tensor along dimension 0."""
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reversed__, (self,), self)
        if self.dim() == 0:
            return self
        else:
            return self.flip(0)

    def norm(
        self,
        p: Optional[Union[float, str]] = "fro",
        dim=None,
        keepdim=False,
        dtype=None,
    ):
        r"""See :func:`torch.norm`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.norm, (self,), self, p=p, dim=dim, keepdim=keepdim, dtype=dtype
            )
        return torch.norm(self, p, dim, keepdim, dtype=dtype)

    def solve(self, other):
        from torch._linalg_utils import solve

        return solve(self, other)

    def lstsq(self, other):
        from torch._linalg_utils import lstsq

        return lstsq(self, other)

    def eig(self, eigenvectors=False):
        from torch._linalg_utils import eig

        return eig(self, eigenvectors=eigenvectors)

    def symeig(self, eigenvectors=False):
        from torch._linalg_utils import _symeig

        return _symeig(self, eigenvectors=eigenvectors)

    def lu(self, pivot=True, get_infos=False):
        r"""See :func:`torch.lu`"""
        # If get_infos is True, then we don't need to check for errors and vice versa
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.lu, (self,), self, pivot=pivot, get_infos=get_infos
            )

        LU, pivots, infos = torch._lu_with_info(
            self, pivot=pivot, check_errors=(not get_infos)
        )
        if get_infos:
            return LU, pivots, infos
        else:
            return LU, pivots

    def stft(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "Optional[Tensor]" = None,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: Optional[bool] = None,
        return_complex: Optional[bool] = None,
        align_to_window: Optional[bool] = None,
    ):
        r"""See :func:`torch.stft`

        .. warning::
          This function changed signature at version 0.4.1. Calling with
          the previous signature may cause error or return incorrect result.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.stft,
                (self,),
                self,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
                return_complex=return_complex,
                align_to_window=align_to_window,
            )
        return torch.stft(
            self,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex=return_complex,
            align_to_window=align_to_window,
        )

    def istft(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "Optional[Tensor]" = None,
        center: bool = True,
        normalized: bool = False,
        onesided: Optional[bool] = None,
        length: Optional[int] = None,
        return_complex: bool = False,
    ):
        r"""See :func:`torch.istft`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.istft,
                (self,),
                self,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                normalized=normalized,
                onesided=onesided,
                length=length,
                return_complex=return_complex,
            )
        return torch.istft(
            self,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            normalized,
            onesided,
            length,
            return_complex=return_complex,
        )

    def resize(self, *sizes):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.resize, (self,), self, *sizes)
        warnings.warn("non-inplace resize is deprecated", stacklevel=2)
        from torch.autograd._functions import Resize

        return Resize.apply(self, sizes)

    def resize_as(self, tensor):
        if has_torch_function_variadic(self, tensor):
            return handle_torch_function(Tensor.resize_as, (self, tensor), self, tensor)
        warnings.warn("non-inplace resize_as is deprecated", stacklevel=2)
        from torch.autograd._functions import Resize

        return Resize.apply(self, tensor.size())

    def split(self, split_size, dim=0):
        r"""See :func:`torch.split`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.split, (self,), self, split_size, dim=dim
            )
        if isinstance(split_size, Tensor):
            try:
                split_size = int(split_size)
            except ValueError:
                pass

        if isinstance(split_size, (int, torch.SymInt)):
            return torch._VF.split(self, split_size, dim)  # type: ignore[attr-defined]
        else:
            return torch._VF.split_with_sizes(
                self,
                # pyrefly: ignore [bad-argument-type]
                split_size,
                dim,
            )

    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        r"""Returns the unique elements of the input tensor.

        See :func:`torch.unique`
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.unique,
                (self,),
                self,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
        return torch.unique(
            self,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )

    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        r"""Eliminates all but the first element from every consecutive group of equivalent elements.

        See :func:`torch.unique_consecutive`
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.unique_consecutive,
                (self,),
                self,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
        return torch.unique_consecutive(
            self, return_inverse=return_inverse, return_counts=return_counts, dim=dim
        )

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rsub__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
        return _C._VariableFunctions.rsub(self, other)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rdiv__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
        return self.reciprocal() * other

    __rtruediv__ = __rdiv__
    __itruediv__ = _C.TensorBase.__idiv__

    # pyrefly: ignore [bad-override]
    __pow__ = cast(
        Callable[
            ["torch._C.TensorBase", Union["Tensor", int, float, bool, complex]],
            "Tensor",
        ],
        _handle_torch_function_and_wrap_type_error_to_not_implemented(
            _C.TensorBase.pow
        ),
    )

    __ipow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(
        _C.TensorBase.pow_
    )

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmod__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
        return torch.remainder(other, self)

    def __format__(self, format_spec):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__format__, (self,), self, format_spec)
        if self.dim() == 0 and not self.is_meta and type(self) is Tensor:
            # Use detach() here to avoid the warning when converting a scalar Tensor that
            # requires gradients to a python number. It is ok for formatting.
            return self.detach().item().__format__(format_spec)
        return object.__format__(self, format_spec)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rpow__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
        return torch.pow(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __floordiv__(self, other: Union["Tensor", int, float, bool]) -> "Tensor":  # type: ignore[override]
        # TODO(rec): the superclass says it accepts complex here,
        # but torch.floor_divide says it doesn't.
        return torch.floor_divide(self, other)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rfloordiv__(self, other: Union["Tensor", int, float, bool]) -> "Tensor":  # type: ignore[override]
        return torch.floor_divide(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rlshift__(
        self, other: Union["Tensor", int, float, bool, complex]
    ) -> "Tensor":
        return torch.bitwise_left_shift(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rrshift__(
        self, other: Union["Tensor", int, float, bool, complex]
    ) -> "Tensor":
        return torch.bitwise_right_shift(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmatmul__(self, other: "Tensor") -> "Tensor":
        return torch.matmul(other, self)

    __pos__ = _C.TensorBase.positive
    __neg__ = _C.TensorBase.neg
    __abs__ = _C.TensorBase.abs

    def __len__(self):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__len__, (self,), self)
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        if torch._C._get_tracing_state():
            warnings.warn(
                "Using len to get tensor shape might cause the trace to be incorrect. "
                "Recommended usage would be tensor.shape[0]. "
                "Passing a tensor of different shape might lead to errors or silently give "
                "incorrect results.",
                category=torch.jit.TracerWarning,
                stacklevel=2,
            )
        return self.shape[0]

    def __iter__(self):
        # NB: we use 'imap' and not 'map' here, so that in Python 2 we get a
        # generator and don't eagerly perform all the indexes.  This could
        # save us work, and also helps keep trace ordering deterministic
        # (e.g., if you zip(*hiddens), the eager map will force all the
        # indexes of hiddens[0] before hiddens[1], while the generator
        # map will interleave them.)
        # NB: We have intentionally skipped __torch_function__ dispatch here.
        # See gh-54457
        if self.dim() == 0:
            raise TypeError("iteration over a 0-d tensor")
        if torch._C._get_tracing_state():
            warnings.warn(
                "Iterating over a tensor might cause the trace to be incorrect. "
                "Passing a tensor of different shape won't change the number of "
                "iterations executed (and might lead to errors or silently give "
                "incorrect results).",
                category=torch.jit.TracerWarning,
                stacklevel=2,
            )
        return iter(self.unbind(0))

    def __hash__(self):
        # Do NOT handle __torch_function__ here as user's default
        # implementation that handle most functions will most likely do it wrong.
        # It can be easily overridden by defining this method on the user
        # subclass if needed.
        return id(self)

    def __dir__(self):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dir__, (self,), self)
        tensor_methods = dir(self.__class__)
        tensor_methods.remove("volatile")  # deprecated
        attrs = list(self.__dict__.keys())
        keys = tensor_methods + attrs

        # property only available dense, cuda tensors
        if (not self.is_cuda) or self.is_sparse:
            keys.remove("__cuda_array_interface__")

        return sorted(keys)

    # Numpy array interface, to support `numpy.asarray(tensor) -> ndarray`
    __array_priority__ = 1000  # prefer Tensor ops over numpy ones

    def __array__(self, dtype=None):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__array__, (self,), self, dtype=dtype)
        if dtype is None
```



## High-Level Overview


This Python file contains 13 class(es) and 69 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Tensor`

**Functions defined**: `_handle_torch_function_and_wrap_type_error_to_not_implemented`, `wrapped`, `_rebuild_from_type`, `_rebuild_from_type_v2`, `_dtype_to_typestr`, `_clear_non_serializable_cached_data`, `__deepcopy__`, `__reduce_ex__`, `storage`, `_typed_storage`, `_reduce_ex_internal`, `__setstate__`, `__repr__`, `backward`, `index`, `register_hook`, `register_post_accumulate_grad_hook`, `reinforce`, `trim`, `is_shared`

**Key imports**: copyreg, enum, functools, itertools, warnings, OrderedDict, Callable, deepcopy, Number, Any, cast, Concatenate, Optional, TypeVar, Union


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copyreg`
- `enum`
- `functools`
- `itertools`
- `warnings`
- `collections`: OrderedDict
- `collections.abc`: Callable
- `copy`: deepcopy
- `numbers`: Number
- `typing`: Any, cast, Concatenate, Optional, TypeVar, Union
- `typing_extensions`: ParamSpec
- `torch`
- `torch._C as _C`
- `torch.utils.hooks`: warn_if_has_hooks
- `functorch.dim`: index
- `torch._linalg_utils`: solve
- `torch.autograd._functions`: Resize
- `torch.fx.experimental.symbolic_shapes`: guard_or_false
- `torch._prims_common as utils`
- `torch_xla`
- `torch_xla.utils.dlpack as xla_dlpack`
- `torch.utils.dlpack`: DLDeviceType


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_tensor_docs.py_docs.md`](./_tensor_docs.py_docs.md)
- [`_classes.py_docs.md`](./_classes.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`_meta_registrations.py_docs.md`](./_meta_registrations.py_docs.md)
- [`_appdirs.py_docs.md`](./_appdirs.py_docs.md)
- [`_streambase.py_docs.md`](./_streambase.py_docs.md)
- [`_lowrank.py_docs.md`](./_lowrank.py_docs.md)
- [`_size_docs.py_docs.md`](./_size_docs.py_docs.md)


## Cross-References

- **File Documentation**: `_tensor.py_docs.md`
- **Keyword Index**: `_tensor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
