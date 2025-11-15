# Documentation: `torch/onnx/_internal/fx/passes/type_promotion.py`

## File Metadata

- **Path**: `torch/onnx/_internal/fx/passes/type_promotion.py`
- **Size**: 64,728 bytes (63.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Owner(s): ["module: onnx"]
from __future__ import annotations

import abc
import dataclasses
import inspect
import logging
from typing import Any, TYPE_CHECKING

import torch
import torch._dispatch.python
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    wrappers as _prims_common_wrappers,
)
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch.fx.experimental import proxy_tensor
from torch.onnx._internal.fx import _pass, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

    from torch._subclasses import fake_tensor


logger = logging.getLogger(__name__)


def _try_getclosurevars(func):
    try:
        return inspect.getclosurevars(func)
    except TypeError:
        return None


@dataclasses.dataclass
class TypePromotionSnapshot:
    """Type promotion snapshot for a fx node and its inputs.

    Contains the promoted dtype for args and kwargs that needs promoting.
    Contains the expected node output dtype.
    """

    args_dtypes: Mapping[int, torch.dtype]
    """Mapping from arg position to dtype to promote to."""

    kwargs_dtypes: Mapping[str, torch.dtype]
    """Mapping from kwarg name to dtype to promote to."""

    out_dtype: torch.dtype
    """Expected output dtype of the node."""


class TypePromotionRule(abc.ABC):
    """Base class for type promotion rule per 'torch.ops.{namespace}.{op_name}'."""

    def __init__(self, namespace: str, op_name: str) -> None:
        self.namespace = namespace
        self.op_name = op_name

    # Make this abstract as well because subclass needs to override __eq__().
    # A class that overrides __eq__() and does not define __hash__() will have its __hash__() implicitly set to None.
    # Ref: https://docs.python.org/3/reference/datamodel.html#object.__hash__
    @abc.abstractmethod
    def __hash__(self) -> int: ...

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool: ...

    def is_valid(self) -> bool:
        """Check if the rule is valid."""
        # This always returns a module. If the module does not exist it will be created.
        module = getattr(torch.ops, self.namespace)
        py_op = getattr(module, self.op_name, None)
        if py_op is None:
            logger.warning(
                "Cannot find op: %s in module: %s", self.op_name, self.namespace
            )
            return False
        if not isinstance(py_op, torch._ops.OpOverloadPacket):
            logger.warning(
                "Op: torch.ops.%s.%s is not an OpOverloadPacket, got: %s",
                self.namespace,
                self.op_name,
                type(py_op),
            )
            return False

        return True

    @abc.abstractmethod
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        """Preview type promotion results for provided set of args and kwargs.

        Returns a TypePromotionSnapshot object that contains the promoted dtypes for
        the arguments and the expected output dtype.
        """
        ...


class ElementwiseTypePromotionRule(TypePromotionRule):
    """Defines how to perform elementwise type promotion for 'torch.ops.{namespace}.{op_name}'."""

    _USE_OPMATH: bool = False
    """Whether to use opmath to compute the promoted input dtype.
    If used, upcasts will be inserted everywhere for lower precision models.
    Set to False and have torchlib handle upcasts in op implementation internally.
    """

    def __init__(
        self,
        namespace: str,
        op_name: str,
        promote_args_positions: Sequence[int],
        promote_kwargs_names: Sequence[str],
        promotion_kind: _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND,
    ) -> None:
        """Constructs a TypePromotionRule for elementwise operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.add'.
            op_name: Name of the op. E.g. 'add' in 'torch.ops.aten.add'.
            promote_args_positions: Positions of args to promote.
            promote_kwargs_names: Names of kwargs to promote.
            promotion_kind: Type promotion kind. Refer to [_prims_common.elementwise_dtypes](https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promote_args_positions = promote_args_positions
        self.promote_kwargs_names = promote_kwargs_names
        self.promotion_kind = promotion_kind

    def __repr__(self) -> str:
        return (
            f"ElementwiseTypePromotionRule('{self.namespace}', '{self.op_name}', "
            f"{self.promote_args_positions}, {self.promote_kwargs_names}, {self.promotion_kind})"
        )

    # pyrefly: ignore [bad-override]
    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, ElementwiseTypePromotionRule):
            return False
        return (
            self.namespace == other.namespace
            and self.op_name == other.op_name
            and self.promote_args_positions == other.promote_args_positions
            and self.promote_kwargs_names == other.promote_kwargs_names
            and self.promotion_kind == other.promotion_kind
        )

    def __hash__(self) -> int:
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def _consolidate_input_dtype(
        self, computed_dtype: torch.dtype, result_dtype: torch.dtype
    ) -> torch.dtype:
        """
        Although opmath is the right thing to do to retain on-par precision, it inserts
        upcasts everywhere in the graph. This is particularly hard for backend to optimize
        since there is no way to differentiate between inserted upcasts and model code
        casts. Hence we consolidate the input dtype to the result dtype to avoid this.
        """
        if not self._USE_OPMATH and self.promotion_kind in (
            _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        ):
            return result_dtype
        return computed_dtype

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        candidate_args = {
            i: args[i]
            for i in self.promote_args_positions
            if i < len(args) and args[i] is not None
        }
        candidate_kwargs = {
            name: kwargs[name]
            for name in self.promote_kwargs_names
            if name in kwargs and kwargs[name] is not None
        }

        computed_dtype, result_dtype = _prims_common.elementwise_dtypes(
            *_pytree.arg_tree_leaves(*candidate_args.values(), **candidate_kwargs),
            type_promotion_kind=self.promotion_kind,
        )

        consolidated_input_dtype = self._consolidate_input_dtype(
            computed_dtype, result_dtype
        )

        return TypePromotionSnapshot(
            dict.fromkeys(candidate_args.keys(), consolidated_input_dtype),
            dict.fromkeys(candidate_kwargs.keys(), consolidated_input_dtype),
            result_dtype,
        )


class DivElementwiseTypePromotionRule(ElementwiseTypePromotionRule):
    """Reference type promotion rule from torch._refs.div.

    Rule depends on the value of the `rounding_mode` argument.
    """

    def __init__(self) -> None:
        super().__init__(
            "aten",
            "div",
            promote_args_positions=(0, 1),
            promote_kwargs_names=(),
            promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        rounding_mode = kwargs.get("rounding_mode")
        if rounding_mode is None:
            # true_divide
            self.promotion_kind = (
                _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
            )
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "trunc":
            # trunc_divide
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "floor":
            # floor_divide
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        raise ValueError(f"Unknown rounding_mode: {rounding_mode}")


class ReductionTypePromotionRule(TypePromotionRule):
    def __init__(
        self,
        namespace: str,
        op_name: str,
        promotion_kind: _prims_common.REDUCTION_OUTPUT_TYPE_KIND,
    ) -> None:
        """Constructs a TypePromotionRule for reduction operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.sum'.
            op_name: Name of the op. E.g. 'sum' in 'torch.ops.aten.sum'.
            promotion_kind: Type promotion kind. Refer to [_prims_common.reduction_dtypes]((https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py)) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promotion_kind = promotion_kind

    def __repr__(self) -> str:
        return f"ReductionTypePromotionRule('{self.namespace}', '{self.op_name}', {self.promotion_kind})"

    # pyrefly: ignore [bad-override]
    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, ElementwiseTypePromotionRule):
            return False
        return (
            self.namespace == other.namespace
            and self.op_name == other.op_name
            and self.promotion_kind == other.promotion_kind
        )

    def __hash__(self) -> int:
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        assert len(args) >= 1, (
            f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
        )
        arg = args[0]
        assert isinstance(arg, torch.Tensor), f"{type(arg)=} is not torch.Tensor"
        dtype: torch.dtype | None = kwargs.get("dtype")

        computation_dtype, result_dtype = _prims_common.reduction_dtypes(
            arg, self.promotion_kind, dtype
        )
        if result_dtype is None:
            # Inspecting code, this can only happen when `promotion_kind` is `KEEP_PROMOTED_TYPE`.
            # Hence set same as computation_dtype.
            result_dtype = computation_dtype

        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )


class AllOrAnyReductionTypePromotionRule(ReductionTypePromotionRule):
    """Reference type promotion rule from torch.ops.aten.all or torch.ops.aten.any.

    This is a special case where computation dtype is always torch.bool.
    The result dtype is always uint8 if `dtype` kwarg is uint8, otherwise torch.bool.
    """

    def __init__(self, op_name: str) -> None:
        super().__init__(
            "aten",
            op_name,
            _prims_common.REDUCTION_OUTPUT_TYPE_KIND.ALWAYS_BOOL,
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        assert len(args) >= 1, (
            f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
        )
        arg = args[0]
        assert isinstance(arg, torch.Tensor), f"{type(arg)=} is not torch.Tensor"
        computation_dtype = torch.bool
        # Preserves uint8 -- probably a legacy mask thing
        result_dtype = torch.uint8 if arg.dtype == torch.uint8 else torch.bool
        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )


class SumLikeReductionTypePromotionRule(ReductionTypePromotionRule):
    """Reference type promotion rule from torch.ops.aten.sum.

    This is a special case where computation dtype is always torch.int64 for integral arg,
    unless overridden by `dtype` kwarg.
    """

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        assert len(args) >= 1, (
            f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
        )
        arg = args[0]
        assert isinstance(arg, torch.Tensor), f"{type(arg)=} is not torch.Tensor"
        dtype: torch.dtype | None = kwargs.get("dtype")
        # The below logic is copied from `torch/_refs/__init__.py` reduction ops impl.
        if dtype is None:
            if _prims_common.is_boolean_dtype(
                arg.dtype
            ) or _prims_common.is_integer_dtype(arg.dtype):
                dtype = torch.int64
            else:
                dtype = arg.dtype
        return super().preview_type_promotion(args, {"dtype": dtype})


# NOTE: [Update type promotion rule]
# BELOW TABLE IS GENERATED FROM `TypePromotionRuleSetGenerator.generate_from_torch_refs`.
# DO NOT EDIT MANUALLY !!!
# For missing rules or discrepancies, please
# 1. Run `pytest test/onnx/test_fx_type_promotion.py` to validate if the generated rule set is current.
#    If it is not, update with new generated set.
# 2. If discrepancies still exist, consider debugging torch._refs or report a bug.
# 3. If rules are still missing, add them to `_EXTRA_TYPE_PROMOTION_RULE_SET` or report a bug.
# Check `TypePromotionRule` class for how each rule is defined and used.
_GENERATED_ATEN_TYPE_PROMOTION_RULE_SET = {
    ElementwiseTypePromotionRule(
        "aten", "abs", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "abs_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "add", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "add_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcdiv", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcdiv_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcmul", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcmul_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addr", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan2_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cat", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "cauchy", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cauchy_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ceil", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ceil_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "celu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "celu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "clamp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "clamp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "copysign", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "copysign_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "deg2rad", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "deg2rad_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "digamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "digamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "dot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "elu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "elu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "eq", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "eq_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "erf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erf_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfinv", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfinv_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "expm1", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "expm1_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exponential", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exponential_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fill", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmax", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmin", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmod", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmod_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "frac", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "frac_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gcd", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gcd_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ge", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "ge_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "gelu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "geometric", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "geometric_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "glu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "gt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "hardtanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "heaviside", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "heaviside_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "huber_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "hypot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "hypot_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "i0", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "i0_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igamma", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igamma_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igammac", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igammac_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "isfinite", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isnan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isneginf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isposinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isreal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "l1_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lcm", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lcm_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "le", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "le_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "leaky_relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lerp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lerp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lgamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lgamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log10", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log10_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log1p", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log1p_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_normal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_normal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logaddexp", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logaddexp2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logsumexp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "lt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "maximum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "minimum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mish", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mish_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mse_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mul", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mul_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ne", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "ne_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "neg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "neg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "nextafter", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "nextafter_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "nll_loss", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "normal", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "pdist", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "poisson_nll_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "prelu", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rad2deg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rad2deg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "reciprocal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "reciprocal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "remainder", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "remainder_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "round", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rsqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rsqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "selu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "selu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sgn", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sgn_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sigmoid", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sigmoid_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sign", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sign_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "signbit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "sin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "smooth_l1_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "softplus", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "square", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "square_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "sub", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sub_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "threshold", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "threshold_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "true_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "true_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "trunc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "trunc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "vdot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "where", [1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "xlogy", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "xlogy_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
}

# Manually curated extra type promotion rules. Please see NOTE [Update type promotion rule]
# before adding new rules.
_EXTRA_TYPE_PROMOTION_RULE_SET = {
    # torch._refs skips type promotion decoration for `clamp_min` and `clamp_max` since
    # the call is routed to the decorated `aten.clamp` op.
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_max",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_min",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    # torch.ops.aten.div.Tensor_mode applies different type promotion rules
    # depending on the value of the `mode` argument.
    DivElementwiseTypePromotionRule(),
    # Manually curating reduction ops since the logic is written inside the op reference
    # implementation.
    AllOrAnyReductionTypePromotionRule("all"),
    AllOrAnyReductionTypePromotionRule("any"),
    ReductionTypePromotionRule(
        "aten",
        "amax",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    ReductionTypePromotionRule(
        "aten",
        "amin",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    # torch.ops.aten.mean is a special case that does not need type promotion.
    ReductionTypePromotionRule(
        "aten",
        "std",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "std_mean",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "var",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumprod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumsum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "prod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "sum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
}


class ElementwiseTypePromotionRuleSetGenerator:
    """Hackly distilling info from reference ops decorated with elementwise type promotion rule.

    The goal is to retrieve the decorator

    ```python
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a", "b"),
            type_promotion_kind=type_promotion_kind,
        )
    ```

    from the reference ops. It provides info as for which arguments are promoted
    and what kind of promotion is applied.
    """

    @classmethod
    def generate_from_torch_refs(cls) -> set[ElementwiseTypePromotionRule]:
        """Parse type promotion rules from reference ops under torch._C._refs."""
        rule_set = set()
        rule_set.update(cls._parse_torch_refs(_refs))
        rule_set.update(cls._parse_torch_refs(_nn_refs))
        rule_set.update(cls._parse_torch_refs(_linalg_refs))
        rule_set.update(cls._parse_torch_refs(_special_refs))
        rule_set.update(cls._parse_torch_refs(_functional_refs))
        return rule_set

    @classmethod
    def _parse_torch_refs(
        cls, ref_module: ModuleType
    ) -> set[ElementwiseTypePromotionRule]:
        logger.info("Processing module: %s", ref_module.__name__)
        rule_set = set()
        for name in ref_module.__all__:
            decorated_op = getattr(ref_module, name)
            rule = cls._parse_type_promotion_rule_from_refs_op(decorated_op)
            if rule is not None and rule.is_valid():
                rule_set.add(rule)

        return rule_set

    @classmethod
    def _parse_type_promotion_rule_from_refs_op(
        cls,
        decorated_op: Callable,
    ) -> ElementwiseTypePromotionRule | None:
        """Retrieve and parse type promotion decorator from op under torch._refs."""
        fn = decorated_op
        type_promo_wrapper = None
        while fn_closure_vars := _try_getclosurevars(fn):
            if "fn" not in fn_closure_vars.nonlocals:
                break
            if "self" in fn_closure_vars.nonlocals and isinstance(
                fn_closure_vars.nonlocals["self"],
                _prims_common_wrappers.elementwise_type_promotion_wrapper,
            ):
                type_promo_wrapper = fn_closure_vars.nonlocals["self"]
                break
            fn = fn_closure_vars.nonlocals["fn"]

        if type_promo_wrapper is not None:
            signature = inspect.signature(decorated_op)

            pos = 0
            promote_args_positions = []
            promote_kwargs_names = []

            if type_promo_wrapper.type_promoting_arg_names is not None:
                for name, param in signature.parameters.items():
                    if name in type_promo_wrapper.type_promoting_arg_names:
                        if param.kind in (
                            param.POSITIONAL_OR_KEYWORD,
                            param.POSITIONAL_ONLY,
                        ):
                            promote_args_positions.append(pos)
                        elif param.kind == param.KEYWORD_ONLY:
                            promote_kwargs_names.append(name)
                    pos += 1

            return ElementwiseTypePromotionRule(
                "aten",
                decorated_op.__name__,
                promote_args_positions=promote_args_positions,
                promote_kwargs_names=promote_kwargs_names,
                promotion_kind=type_promo_wrapper.type_promotion_kind,
            )

        logger.warning(
            "Cannot find type promotion rule for: %s.%s",
            decorated_op.__module__,
            decorated_op.__name__,
        )
        return None


class TypePromotionTable:
    """Type promotion table for torch.ops."""

    def __init__(self) -> None:
        self._rule_table = {}
        for rule in _GENERATED_ATEN_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)
        for rule in _EXTRA_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)

    def add_rule(self, rule: TypePromotionRule) -> None:
        """Add a type promotion rule for a python op in a torch.ops module.

        Args:
            rule: Type promotion rule.
            module: Module containing the op. E.g. torch.ops.aten.

        Raises:
            ValueError: If the rule is invalid.
        """
        if not rule.is_valid():
            raise ValueError(f"Invalid type promotion rule: {rule}")
        self._rule_table[f"{rule.namespace}.{rule.op_name}"] = rule

    def get_rule(self, py_op: torch._ops.OpOverloadPacket) -> TypePromotionRule | None:
        """Get type promotion rule for a python op under 'torch.ops.<namespace>'."""
        return self._rule_table.get(str(py_op), None)


def get_type_promotion_rule(
    node: torch.fx.Node,
    type_promotion_table: TypePromotionTable,
) -> TypePromotionRule | None:
    """Get type promotion rule for a node.

    Args:
        node: Node to get type promotion rule for.
        type_promotion_table: Type promotion table.

    Returns:
        Type promotion rule for the node. None if no rule is found or if the node is not
        representing a torch operator.
    """
    op = node.target
    if not isinstance(op, torch._ops.OpOverload):
        return None
    if (rule := type_promotion_table.get_rule(op.overloadpacket)) is None:
        return None

    return rule


class _OpTraceDispatchMode(_python_dispatch.TorchDispatchMode):
    """Trace ops that were dispatched.

    Utilize the dispatch mechanism in [`__torch_dispatch__`](https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557)
    to trace op overloads that were dispatched to. This is used to find the compatible
    op overload for a given op overload packet for different set of args and kwargs.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.traced_ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.traced_ops.append(func)
        return func(*args, **kwargs)


def find_compatible_op_overload(
    op: torch._ops.OpOverloadPacket, args: tuple, kwargs: dict
) -> torch._ops.OpOverload:
    """Find compatible OpOverload for an OpOverloadPacket using provided args and kwargs.

    Each "call_function" fx.Node in the fx.GraphModule has a target that represents a torch._ops.OpOverload.
    The OpOverload contains an OpOverloadPacket that holds all the available overloads for the operation.

    During the type promotion pass, there are cases where the types of the args and kwargs may change,
    such as promoting Python numbers to tensors. Consequently, the original OpOverload might not be
    compatible with the updated args and kwargs. This function is used to identify the compatible
    OpOverload for the given args and kwargs.

    Args:
        op: OpOverloadPacket to find compatible OpOverload for.
        args: The positional arguments to consider for compatibility.
        kwargs: The keyword arguments to consider for compatibility.

    Returns:
        torch._ops.OpOverload: The compatible OpOverload found for the given args and kwargs.

    Raises:
        RuntimeError: If no compatible op overload is found.

    Examples:
        >>> import torch
        >>> packet = torch.ops.aten.pow
        >>> args = (torch.tensor([1.0, 2.0]), 2)
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Scalar'
        >>> args = (torch.tensor([1.0, 2.0]), torch.tensor(2.0))
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Tensor'
    """
    # Utilize the dispatch mechanism to find the compatible op overload.
    op_trace_dispatch_mode = _OpTraceDispatchMode()
    with op_trace_dispatch_mode:
        op(
```



## High-Level Overview

"""Type promotion snapshot for a fx node and its inputs.    Contains the promoted dtype for args and kwargs that needs promoting.    Contains the expected node output dtype.

This Python file contains 17 class(es) and 43 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TypePromotionSnapshot`, `TypePromotionRule`, `ElementwiseTypePromotionRule`, `DivElementwiseTypePromotionRule`, `ReductionTypePromotionRule`, `AllOrAnyReductionTypePromotionRule`, `SumLikeReductionTypePromotionRule`, `ElementwiseTypePromotionRuleSetGenerator`, `TypePromotionTable`, `_OpTraceDispatchMode`, `_TypePromotionInterpreter`, `InsertTypePromotion`

**Functions defined**: `_try_getclosurevars`, `__init__`, `__hash__`, `__repr__`, `__eq__`, `is_valid`, `preview_type_promotion`, `__init__`, `__repr__`, `__eq__`, `__hash__`, `_consolidate_input_dtype`, `preview_type_promotion`, `__init__`, `preview_type_promotion`, `__init__`, `__repr__`, `__eq__`, `__hash__`, `preview_type_promotion`

**Key imports**: annotations, abc, dataclasses, inspect, logging, Any, TYPE_CHECKING, torch, torch._dispatch.python, torch._ops, torch.fx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/fx/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `abc`
- `dataclasses`
- `inspect`
- `logging`
- `typing`: Any, TYPE_CHECKING
- `torch`
- `torch._dispatch.python`
- `torch._ops`
- `torch.fx`
- `torch.fx.traceback as fx_traceback`
- `torch._refs`: linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
- `torch._refs.nn`: functional as _functional_refs
- `torch.fx.experimental`: proxy_tensor
- `torch.onnx._internal.fx`: _pass, type_utils as fx_type_utils
- `torch.utils`: _python_dispatch, _pytree
- `collections.abc`: Callable, Mapping, Sequence
- `types`: ModuleType
- `torch._subclasses`: fake_tensor


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/onnx/_internal/fx/passes`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `type_promotion.py_docs.md`
- **Keyword Index**: `type_promotion.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
