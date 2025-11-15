# Documentation: `docs/torch/ao/quantization/fx/prepare.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/fx/prepare.py_docs.md`
- **Size**: 53,899 bytes (52.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/fx/prepare.py`

## File Metadata

- **Path**: `torch/ao/quantization/fx/prepare.py`
- **Size**: 90,293 bytes (88.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copy
import warnings
from dataclasses import asdict
from typing import Any

import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization import (
    _DerivedObserverOrFakeQuantize,
    FixedQParamsFakeQuantize,
    FixedQParamsObserver,
    ObserverBase,
    ObserverOrFakeQuantize,
    PlaceholderObserver,
)
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeConfig,
    get_native_backend_config,
)
from torch.ao.quantization.backend_config.utils import (
    get_fusion_pattern_to_root_node_getter,
    get_module_to_qat_module,
    get_pattern_to_dtype_configs,
)
from torch.ao.quantization.observer import _is_activation_post_process, _PartialWrapper
from torch.ao.quantization.qconfig import _is_reuse_input_qconfig, QConfigAny
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize import convert, propagate_qconfig_
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    EdgeOrNode,
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)
from torch.ao.quantization.utils import (
    _parent_name,
    get_qconfig_dtypes,
    get_swapped_custom_module_class,
    NodePattern,
    Pattern,
)
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from torch.fx.node import Argument

from ._equalize import is_equalization_observer, node_supports_equalization
from .custom_config import PrepareCustomConfig, StandaloneModuleConfigEntry
from .match_utils import _find_matches, _MatchResultWithQConfig
from .pattern_utils import _sorted_patterns_dict
from .qconfig_mapping_utils import (
    _generate_node_name_to_qconfig,
    _get_flattened_qconfig_dict,
    _update_qconfig_for_fusion,
    _update_qconfig_for_qat,
)
from .quantize_handler import (
    _default_root_node_getter,
    _get_pattern_to_quantize_handlers,
    QuantizeHandler,
)
from .utils import (
    _insert_dequant_stubs_for_custom_module_lstm_output,
    _is_custom_module_lstm,
    _maybe_get_custom_module_lstm_from_node_arg,
    _qconfig_satisfies_dtype_config_constraints,
    all_node_args_have_no_tensors,
    assert_and_get_unique_device,
    get_custom_module_class_keys,
    get_new_attr_name_with_prefix,
    get_non_observable_arg_indexes_and_types,
    node_arg_is_bias,
    node_arg_is_weight,
    NON_QUANTIZABLE_WEIGHT_OPS,
    ObservedGraphModuleAttrs,
)


__all__ = [
    "insert_observers_for_model",
    "prepare",
    "propagate_dtypes_for_known_nodes",
]


# list of dtypes to not add observers to
_DO_NOT_OBS_DTYPE_LIST = [int, float, torch.bool, None]
_OBS_DTYPE_LIST = [
    torch.quint8,
    torch.qint8,
    torch.qint32,
    torch.float16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.float8_e5m2,
    torch.float8_e4m3fn,
]

_DEFAULT_FP32_OBS_OR_FQ_CTR = PlaceholderObserver.with_args(dtype=torch.float)

# note: the following default target dtype info dicts are temporary,
# should be moved to the new programmable API class soon
_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO = {
    "input_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig.activation,
    "output_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig.activation,
}

_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO = {
    "input_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_quint8_placeholder_qconfig.activation,
    "output_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_quint8_placeholder_qconfig.activation,
}


def _get_observer_kwargs(
    quant_spec: QuantizationSpec | FixedQParamsQuantizationSpec,
):
    kwargs_dict = asdict(quant_spec)
    return copy.deepcopy(kwargs_dict)


def _get_qspec_for_arg(
    arg: Node,
    input_qspec_map: dict[Node, QuantizationSpecBase],
    named_modules: dict[str, torch.nn.Module],
) -> QuantizationSpecBase | None:
    while _is_activation_post_process_node(arg, named_modules):
        arg = arg.args[0]  # type: ignore[assignment]
    return input_qspec_map.get(arg)


def _create_obs_or_fq_from_qspec(
    quantization_spec: QuantizationSpecBase | None,
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
):
    """Create observer or fake quantize objects based on quantization spec

    Args:
       quantization_spec: used to store parameters to create the observer or fake quantizer
       obs_or_fq_map: this is a map from edge/output to the corresponding observer/fake_quant
       instance, it may be reused for different edge/output depending on configuration
    """
    if quantization_spec is None:
        return None
    if isinstance(quantization_spec, SharedQuantizationSpec):
        edge_or_node = quantization_spec.edge_or_node
        if edge_or_node not in obs_or_fq_map:
            raise AssertionError(
                "please make sure only refer to edge or node that has "
                f"observer/fake_quant inserted: '{edge_or_node}' not in\n{obs_or_fq_map.keys()}"
            )
        return obs_or_fq_map[edge_or_node]
    elif isinstance(quantization_spec, DerivedQuantizationSpec):
        # can't use asdict, so not calling get_observer_kwargs here
        kwargs = {
            "dtype": quantization_spec.dtype,
            "derive_qparams_fn": quantization_spec.derive_qparams_fn,
            "quant_min": quantization_spec.quant_min,
            "quant_max": quantization_spec.quant_max,
            "qscheme": quantization_spec.qscheme,
            "ch_axis": quantization_spec.ch_axis,
        }
        edge_or_nodes = quantization_spec.derived_from
        obs_or_fqs = [obs_or_fq_map[k] for k in edge_or_nodes]
        # pyrefly: ignore [unsupported-operation]
        kwargs["obs_or_fqs"] = obs_or_fqs
        return _DerivedObserverOrFakeQuantize.with_args(**kwargs)()
    elif isinstance(quantization_spec, FixedQParamsQuantizationSpec):
        kwargs = _get_observer_kwargs(quantization_spec)
        observer_ctr = FixedQParamsObserver.with_args(**kwargs)
        if is_qat:
            return FixedQParamsFakeQuantize.with_args(observer=observer_ctr)()
        else:
            return observer_ctr()

    if not isinstance(quantization_spec, QuantizationSpec):
        raise AssertionError("quantization_spec must be a QuantizationSpec")
    observer_or_fake_quant_ctr = quantization_spec.observer_or_fake_quant_ctr
    kwargs = _get_observer_kwargs(quantization_spec)
    kwargs.pop("observer_or_fake_quant_ctr")
    # we will remove is_dynamic from QuantizationSpec because
    # it seems that dynamic range quantization
    obs_or_fq_class = observer_or_fake_quant_ctr
    if isinstance(observer_or_fake_quant_ctr, _PartialWrapper):
        obs_or_fq_class = observer_or_fake_quant_ctr.p.func  # type: ignore[union-attr, assignment]
    if "PerChannel" not in obs_or_fq_class.__name__:  # type: ignore[operator, union-attr]
        kwargs.pop("ch_axis")
    return observer_or_fake_quant_ctr.with_args(**kwargs)()


def _needs_obs_or_fq(
    prev_output_dtype: Any,
    prev_output_is_dynamic: bool,
    cur_target_dtype: Any,
    cur_target_is_dynamic: bool,
    reuse_input_obs_or_fq: bool,
    is_zeroth_arg: bool = False,
) -> bool:
    """
    note: we will treat "not specified" as torch.float for now
    utility function that checks if we should insert an observer or fake quant node
    base on the requested dtype for the nodes from user

    is_zeroth_arg: we only dynamically quantize the first arg of the node right now
      this should be removed when we enable configuring dynamic quantization
      for a specific argument, this can be removed if we deprecate fx graph mode
      quantization

    """

    # need to insert placeholder observer for dynamic quantization so that it can
    # be converted to choose_qparams -> q -> dq in convert step
    if cur_target_is_dynamic:
        if cur_target_dtype not in _OBS_DTYPE_LIST:
            raise AssertionError(
                f"Expected cur_target_dtype to be torch.float, but got: {cur_target_dtype}"
            )
        if prev_output_dtype in _DO_NOT_OBS_DTYPE_LIST:
            raise AssertionError(
                "prev_output_dtype must not be in _DO_NOT_OBS_DTYPE_LIST"
            )
        return is_zeroth_arg
    if reuse_input_obs_or_fq:
        return False
    # non dynamic quantization
    if cur_target_dtype in _OBS_DTYPE_LIST:
        return (
            prev_output_dtype in _OBS_DTYPE_LIST + [torch.float]
            and cur_target_dtype != prev_output_dtype
        )

    # lots of error checking are skipped here for now
    return False


def _is_activation_post_process_node(
    node: Node, named_modules: dict[str, torch.nn.Module]
) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_module"
        and _is_activation_post_process(named_modules[str(node.target)])
    )


def _get_dtype_and_is_dynamic(
    obs_or_fq: ObserverOrFakeQuantize | None,
) -> tuple[torch.dtype | None, bool]:
    """Given a constructor for observer or fake quant module, returns
    a Tuple of dtype and is_dynamic
    """
    # TODO: instead of instantiating the instance, we can use inspect to get the default args
    if obs_or_fq is None:
        return None, False
    else:
        return obs_or_fq.dtype, getattr(obs_or_fq, "is_dynamic", False)  # type: ignore[return-value]


def _is_input_arg_dtype_supported_by_backend(
    arg: Argument,
    node: Node,
    qconfig: QConfigAny,
    dtype_config: DTypeConfig,
    backend_config: BackendConfig,
) -> bool:
    """Check if the configured qconfig for the argument
    is supported by the backend or not
    """
    if isinstance(arg, (list, tuple)):
        return all(
            _is_input_arg_dtype_supported_by_backend(
                a, node, qconfig, dtype_config, backend_config
            )
            for a in arg
        )
    if not isinstance(arg, Node):
        return True
    # TODO: support check for standalone module
    is_weight = node_arg_is_weight(node, arg)
    is_bias = node_arg_is_bias(node, arg)
    is_activation = not is_weight and not is_bias
    if is_activation:
        input_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "input_act_obs_or_fq_ctr"
        )
        input_act_obs_or_fq = (
            input_act_obs_or_fq_ctr() if input_act_obs_or_fq_ctr else None
        )
        qconfig_dtype, qconfig_is_dynamic = _get_dtype_and_is_dynamic(
            input_act_obs_or_fq
        )
        # TODO(future PR): remove the cast to bool below after figuring
        # out why backend_config has is_dynamic set to None in some cases.
        return (dtype_config.input_dtype is None) or (
            dtype_config.input_dtype == qconfig_dtype
            and bool(dtype_config.is_dynamic) == bool(qconfig_is_dynamic)
            and _qconfig_satisfies_dtype_config_constraints(
                qconfig, dtype_config.input_dtype_with_constraints
            )
        )
    elif is_weight:
        # TODO: move dtype check into `_qconfig_satisfies_dtype_config_constraints` as well
        weight_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "weight_obs_or_fq_ctr", None
        )
        weight_obs_or_fq = weight_obs_or_fq_ctr() if weight_obs_or_fq_ctr else None
        qconfig_weight_dtype, _ = _get_dtype_and_is_dynamic(weight_obs_or_fq)
        backend_config_weight_dtype = dtype_config.weight_dtype
        dtype_matches = qconfig_weight_dtype == backend_config_weight_dtype
        qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(
            qconfig, dtype_config.weight_dtype_with_constraints, is_activation=False
        )
        return backend_config_weight_dtype is None or (
            dtype_matches and qconfig_satisfies_constraints
        )
    else:  # bias
        # TODO: move dtype check into `_qconfig_satisfies_dtype_config_constraints` as well
        bias_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "bias_obs_or_fq_ctr", None
        )
        bias_obs_or_fq = bias_obs_or_fq_ctr() if bias_obs_or_fq_ctr else None
        qconfig_bias_dtype, _ = _get_dtype_and_is_dynamic(bias_obs_or_fq)
        backend_config_bias_dtype = dtype_config.bias_dtype
        return (
            backend_config_bias_dtype is None
            or qconfig_bias_dtype == backend_config_bias_dtype
        )


def _is_output_dtype_supported_by_backend(
    node: Node,
    qconfig: QConfigAny,
    dtype_config: DTypeConfig,
) -> bool:
    """Check if the configured qconfig for the output
    is supported by the backend or not
    """
    # TODO: move dtype check into `_qconfig_satisfies_dtype_config_constraints` as well
    backend_config_output_dtype = dtype_config.output_dtype
    # TODO: we should check is_dynamic here as well, the code from _is_input_arg_dtype_supported_by_backend
    # from input activation check can be reused here
    qconfig_output_dtype = None
    output_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
        "output_act_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
    )
    output_act_obs_or_fq = (
        output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    )
    qconfig_output_dtype, qconfig_output_is_dynamic = _get_dtype_and_is_dynamic(
        output_act_obs_or_fq
    )
    # TODO: this is a hack because we can only specify one activation_obs_or_fq for
    # qconfig (qconfig.activation), and we are only supporting dynamically quantized
    # linear op which has fp32 output dtype, this should be removed if we generalize
    # the structure of qconfig in the future
    if qconfig_output_is_dynamic:
        qconfig_output_dtype = torch.float32
    dtype_matches = qconfig_output_dtype == backend_config_output_dtype
    qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(
        qconfig, dtype_config.output_dtype_with_constraints
    )
    return backend_config_output_dtype is None or (
        dtype_matches and qconfig_satisfies_constraints
    )


def _is_observer_in_same_graph(
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat,
):
    """Check if observer in same graph
    when the node output is not fp32 and input is 'placeholder'
    the input is assumed to be quantized, so it is observed
    in a different place rather than not observed.
    """
    node_output_dtype = _get_arg_target_dtype_as_output(
        node, named_modules, obs_or_fq_map, is_qat
    )
    if len(node.args) > 0 and isinstance(node.args[0], Node):
        if (
            node_output_dtype in [torch.quint8, torch.uint8]
            and node.args[0].op == "placeholder"
        ):
            return False
    return True


def _is_pattern_dtype_config_and_qconfig_supported_by_backend(
    pattern: Pattern | None,
    matched_node_pattern: list[Node] | None,
    qconfig: QConfigAny,
    backend_config: BackendConfig,
) -> bool:
    """Check if the dtype configuration of a pattern is supported by
    the backend or not, and whether the qconfig satisfies constraints
    specified in the corresponding dtype config.
    """
    if backend_config is None or pattern is None:
        return True
    if matched_node_pattern is None or len(matched_node_pattern) < 1:
        raise AssertionError("matched_node_pattern must be non-empty")
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config)
    dtype_configs: list[DTypeConfig] = pattern_to_dtype_configs.get(pattern, [])
    pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(backend_config)

    root_node_getter = pattern_to_root_node_getter.get(
        pattern, _default_root_node_getter
    )
    root_node = root_node_getter(matched_node_pattern)
    input_node = root_node
    output_node = matched_node_pattern[0]
    for dtype_config in dtype_configs:
        # check if arg dtype are supported
        supported = True
        for arg in list(input_node.args) + list(input_node.kwargs.values()):
            supported = supported and _is_input_arg_dtype_supported_by_backend(
                arg, input_node, qconfig, dtype_config, backend_config
            )
        # check if output dtype is supported
        supported = supported and _is_output_dtype_supported_by_backend(
            output_node, qconfig, dtype_config
        )
        if supported:
            return True
    return False


def _get_standalone_module_configs(
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    prepare_custom_config: PrepareCustomConfig,
    parent_qconfig: QConfigAny,
    parent_backend_config: BackendConfig | None,
) -> tuple[QConfigMapping, tuple[Any, ...], PrepareCustomConfig, BackendConfig | None]:
    """
    Returns the standalone module QConfigMapping and PrepareCustomConfig
    for `node`, assuming that the module pointed to by `node` is
    a standalone modules.
    """
    module_name = str(node.target)
    module_type = type(named_modules[module_name])  # type: ignore[index]
    # name config has precedence over type config
    config_entry = StandaloneModuleConfigEntry(None, (), None, None)
    config_entry = prepare_custom_config.standalone_module_classes.get(
        module_type, config_entry
    )
    config_entry = prepare_custom_config.standalone_module_names.get(
        module_name, config_entry
    )
    # fallback to use parent module's qconfig if user didn't specify qconfig dict
    qconfig_mapping = config_entry.qconfig_mapping or QConfigMapping().set_global(
        parent_qconfig
    )
    example_inputs = config_entry.example_inputs
    prepare_custom_config = config_entry.prepare_custom_config or PrepareCustomConfig()
    backend_config = config_entry.backend_config or parent_backend_config
    return (qconfig_mapping, example_inputs, prepare_custom_config, backend_config)


def _qat_swap_modules(
    root: torch.nn.Module, module_to_qat_module: dict[Pattern, type[torch.nn.Module]]
) -> None:
    convert(root, mapping=module_to_qat_module, inplace=True, remove_qconfig=False)


def _add_matched_node_name_to_set(matched_node_pattern: NodePattern, s: set[str]):
    if isinstance(matched_node_pattern, Node):
        s.add(matched_node_pattern.name)
    elif isinstance(matched_node_pattern, (list, tuple)):
        for maybe_node in matched_node_pattern:
            _add_matched_node_name_to_set(maybe_node, s)


def _insert_obs_or_fq(
    node: Node,
    obs_or_fq: ObserverOrFakeQuantize,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    model_device: torch.device | None = None,
) -> Node:
    """
    Attaches `obs_or_fq` to `model`, and creates a node which calls
    `obs_or_fq` on the output of `node`.

    obs_or_fq: an instance of Observer or FakeQuantize module
    """
    if model_device is None:
        model_device = assert_and_get_unique_device(model)
    if model_device:
        obs_or_fq.to(model_device)
    # add obs_or_fq module as attribute
    if is_equalization_observer(obs_or_fq):
        prefix = node.name + "_equalization_process_"
    else:
        prefix = "activation_post_process_"
    get_new_obs_or_fq_name = get_new_attr_name_with_prefix(prefix)
    obs_or_fq_name = get_new_obs_or_fq_name(model)
    setattr(model, obs_or_fq_name, obs_or_fq)
    named_modules[obs_or_fq_name] = obs_or_fq
    with graph.inserting_after(node):
        new_obs = graph.create_node("call_module", obs_or_fq_name, (node,), {})
    return new_obs


def _set_target_dtype_info_for_matched_node_pattern(
    matched_node_pattern: NodePattern,
    last_node: Node,
    qconfig: QConfigAny,
    qhandler: QuantizeHandler | None,
    backend_config: BackendConfig,
    named_modules: dict[str, torch.nn.Module],
    cache_for_no_tensor_check: dict[Node, bool],
    processed_nodes: set[Node],
) -> None:
    """Sets the target_dtype_info for each node in matched_node_pattern
    Note: processed_nodes is used to ensure we only process each node once
    """
    if isinstance(matched_node_pattern, (list, tuple)):
        for node_pattern in matched_node_pattern:
            _set_target_dtype_info_for_matched_node_pattern(
                node_pattern,
                last_node,
                qconfig,
                qhandler,
                backend_config,
                named_modules,
                cache_for_no_tensor_check,
                processed_nodes,
            )

    # set target_dtype_info if matched_node_pattern is a Node
    # other types of matched object, e.g. int, float literals, are ignored
    elif isinstance(matched_node_pattern, Node):
        # for pyre
        if not isinstance(matched_node_pattern, Node):
            raise AssertionError("matched_node_pattern must be a Node")
        node = matched_node_pattern
        if node in processed_nodes:
            return
        processed_nodes.add(node)

        if qconfig is None:
            return
        # TODO: refactor the following code in terms of apply a qconfig to a pattern
        # e.g. for a pattern with op1 -> op2 -> op3, and qconfig = QConfig(input_act=obs0, output_act=obs1)
        # we set the input_obs_or_fq_ctr for the arguments of op1 to based on qconfig.input_act,
        # and set output_obs_or_fq_ctr based on qconfig.output_act
        # this also requires we extend the structure of QConfig to support more fine
        # grained configurations
        target_dtype_info: dict[str, Any] = _get_target_activation_dtype_for_node(
            node,
            qconfig,
            qhandler,
            named_modules,
            backend_config,
            cache_for_no_tensor_check,
        )
        node.meta["target_dtype_info"] = target_dtype_info


def _get_target_activation_dtype_for_node(
    node: Node,
    qconfig: QConfigAny,
    qhandler: QuantizeHandler | None,
    named_modules: dict[str, torch.nn.Module],
    backend_config: BackendConfig,
    cache_for_no_tensor_check: dict[Node, bool],
) -> dict[str, Any]:
    """
    For each op attribute in the op's input activation, output activation,
    weight, bias - returns the settings of dtype and is_dynamic we expect
    for the `quantize` call in the reference model representation, or None
    if there is no `quantize` call needed.

    For example, if we have a node corresponding to `op0` in

      x0 -> op0 -> x1

    And we want a reference quantized representation to be

      x0 -> quant_static -> dequant -> op0 -> quant_dynamic -> dequant -> x1

    Then this function will return

      {
        "input_act_obs_or_fq_ctr": MinMaxObserver.with_args(dtype=torch.quint8, is_dynamic=False),
        "output_act_obs_or_fq_ctr": MinMaxObserver.with_args(dtype=torch.quint8, is_dynamic=False),
      }

    TODO(future PR, if needed): explicitly spell out the non-Tensor
    dtypes.
    """
    args_have_no_tensors = all_node_args_have_no_tensors(
        node, named_modules, cache_for_no_tensor_check
    )
    if args_have_no_tensors:
        return {
            "input_act_obs_or_fq_ctr": None,
            "output_act_obs_or_fq_ctr": None,
        }
    # get qconfig to determine the eventual dtype of this node
    if qconfig is not None:
        act_dtype, weight_dtype, input_act_is_dynamic = get_qconfig_dtypes(qconfig)

        # Currently `QConfig` only has one `activation` field.
        # For static quantization, it is reused for both input
        # and output activation. For dynamic quantization, this
        # field is currently only used for the input activation,
        # with the output activation being in fp32.
        # In the future this may change as we add more fields
        # to the `QConfig` object.
        bias_dtype = (
            torch.float16
            if (
                act_dtype == torch.float16
                and weight_dtype == torch.float16
                and (not input_act_is_dynamic)
            )
            else torch.float
        )

        is_general_tensor_value_op = (
            qhandler is not None and qhandler.is_general_tensor_value_op()
        )

        _is_standalone_module = qhandler is not None and qhandler.is_standalone_module()

        weight_index = None
        if (
            isinstance(node, Node)
            and node.op == "call_function"
            and node.target in backend_config._pattern_complex_format_to_config
        ):
            weight_index = backend_config._pattern_complex_format_to_config[
                node.target
            ]._input_type_to_index.get("weight")

        bias_index = None
        if (
            isinstance(node, Node)
            and node.op == "call_function"
            and node.target in backend_config._pattern_complex_format_to_config
        ):
            bias_index = backend_config._pattern_complex_format_to_config[
                node.target
            ]._input_type_to_index.get("bias")

        return {
            "input_act_obs_or_fq_ctr": qconfig.activation,
            "weight_obs_or_fq_ctr": qconfig.weight,
            "bias_obs_or_fq_ctr": PlaceholderObserver.with_args(dtype=bias_dtype),
            "weight_index": weight_index,
            "bias_index": bias_index,
            "output_act_obs_or_fq_ctr": qconfig.activation,
            "reuse_input_obs_or_fq": _is_reuse_input_qconfig(qconfig),
            "input_output_share_observers": is_general_tensor_value_op,
            "_is_standalone_module": _is_standalone_module,
        }
    return copy.copy(_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO)


def _get_output_act_obs_or_fq(
    arg: Node,
    named_modules: dict[str, torch.nn.Module],
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> ObserverOrFakeQuantize | None:
    """Get the constructor for observer or fake quant object for
    the argument in the original graph as the output of previous node,
    skipping inserted observers

    We are assuming that the observers are inserted correctly, and the dtype for
    argument in quantized graph will match what is specified by the qconfig
    """
    if not isinstance(arg, Node):
        raise AssertionError("arg must be a Node")
    if "quantization_annotation" in arg.meta:
        return _create_obs_or_fq_from_qspec(
            arg.meta["quantization_annotation"].output_qspec, obs_or_fq_map, is_qat
        )

    # Custom module LSTM output is a tuple that we broke down into the internal nodes in order
    # to insert DeQuantStubs (see `_insert_dequant_stubs_for_custom_module_lstm_output`).
    # Since we modified the graph in this case, we must trace back from the args through
    # the specific nodes we added in order to reach the original LSTM node. Otherwise, we would
    # not be able to accurately detect whether this node is a consumer of custom module LSTM.
    custom_module_lstm_node = _maybe_get_custom_module_lstm_from_node_arg(
        arg, named_modules
    )
    output_act_obs_or_fq_ctr = None
    if custom_module_lstm_node is not None:
        output_act_obs_or_fq_ctr = custom_module_lstm_node.meta["target_dtype_info"][
            "output_act_obs_or_fq_ctr"
        ]
        output_act_obs_or_fq = (
            output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
        )
    elif _is_activation_post_process_node(arg, named_modules):
        observed_arg = arg.args[0]
        if not isinstance(observed_arg, Node):
            raise AssertionError("Currently we only support observing Node")
        if "quantization_annotation" in observed_arg.meta:
            output_act_obs_or_fq = _create_obs_or_fq_from_qspec(
                observed_arg.meta["quantization_annotation"].output_qspec,
                obs_or_fq_map,
                is_qat,
            )
        else:
            if "target_dtype_info" not in observed_arg.meta:
                raise AssertionError(
                    "expected 'target_dtype_info' in observed_arg.meta"
                )
            output_act_obs_or_fq_ctr = observed_arg.meta["target_dtype_info"][
                "output_act_obs_or_fq_ctr"
            ]
            output_act_obs_or_fq = (
                output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
            )
    else:
        if "target_dtype_info" in arg.meta:
            output_act_obs_or_fq_ctr = arg.meta["target_dtype_info"].get(
                "output_act_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
            )
        else:
            output_act_obs_or_fq_ctr = _DEFAULT_FP32_OBS_OR_FQ_CTR
        output_act_obs_or_fq = (
            output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
        )

    return output_act_obs_or_fq


def _get_arg_target_dtype_as_output(
    arg: Node,
    named_modules: dict[str, torch.nn.Module],
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> torch.dtype | None:
    arg_as_output_act_obs_or_fq = _get_output_act_obs_or_fq(
        arg, named_modules, obs_or_fq_map, is_qat
    )
    arg_as_output_target_dtype, _ = _get_dtype_and_is_dynamic(
        arg_as_output_act_obs_or_fq
    )
    return arg_as_output_target_dtype


def _get_arg_as_input_act_obs_or_fq(
    arg: Node,
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> ObserverOrFakeQuantize | None:
    """Get the observer or fake quant constructor for the Argument `arg`, as input
    to Node `node`
    """
    if not isinstance(arg, Node):
        raise AssertionError("arg must be a Node")
    # "input_qspec_map" is the more general design we'll use for pt2e path
    # it is a map from input argument node to observer or fake quant constructor, for example
    # for the following graph:
    # x -> conv -> output
    #
    # we may annotate conv node like the following:
    # conv.meta[...] = QuantizationAnnotation("input_qspec_map": {x: MinMaxObserver.with_args(dtype=torch.qint8)}, ...)
    #
    if "quantization_annotation" in node.meta:
        input_qspec_map = node.meta["quantization_annotation"].input_qspec_map
        input_arg_qspec = _get_qspec_for_arg(arg, input_qspec_map, named_modules)
        if input_arg_qspec is None:
            input_arg_obs_or_fq = _DEFAULT_FP32_OBS_OR_FQ_CTR()
        else:
            input_arg_obs_or_fq = _create_obs_or_fq_from_qspec(
                input_arg_qspec, obs_or_fq_map, is_qat
            )
        return input_arg_obs_or_fq

    # we can remove the following path in the future if fx graph mode quantization is
    # no longer used
    is_weight = node_arg_is_weight(node, arg)
    is_bias = node_arg_is_bias(node, arg)
    is_activation = not is_weight and not is_bias
    obs_or_fq_ctr = None
    if is_activation:
        obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "input_act_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
        )
    elif is_weight:
        if node.target not in NON_QUANTIZABLE_WEIGHT_OPS:
            obs_or_fq_ctr = node.meta["target_dtype_info"].get(
                "weight_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
            )
    else:
        obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "bias_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
        )
    return obs_or_fq_ctr() if obs_or_fq_ctr else None


def _maybe_insert_input_observer_for_arg_or_kwarg(
    node: Node | Any,
    arg: Argument,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    qhandler: QuantizeHandler | None,
    prepare_custom_config: PrepareCustomConfig,
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
    backend_config: BackendConfig | None = None,
    model_device: torch.device | None = None,
) -> Argument:
    """
    Given a `node` and an `arg`, inserts an input observer between
    `node` and `arg` if necessary.
    """
    # for ops such as torch.cat([x0, x1]),
    # traverse through the list
    if isinstance(arg, (list, tuple)):
        new_arg_to_return = []
        for inner_arg in arg:
            new_inner_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
                node,
                inner_arg,
                qconfig,
                model,
                named_modules,
                graph,
                qhandler,
                prepare_custom_config,
                obs_or_fq_map,
                is_qat,
                backend_config,
                model_device,
            )
            new_arg_to_return.append(new_inner_arg)
        return type(arg)(new_arg_to_return)

    if not isinstance(arg, Node):
        return arg
    if not isinstance(arg, Node):
        raise AssertionError("arg must be a Node")
    # default (no observer)
    new_arg = arg

    is_standalone_module = qhandler is not None and qhandler.is_standalone_module()
    # TODO: move this to a separate function
    if not is_standalone_module:
        # Note: qconfig can be None in this branch this we are getting act/fq from
        # node.meta now
        # regular flow for most nodes, except standalone modules

        if "quantization_annotation" in node.meta:
            reuse_input_obs_or_fq = node.meta[
                "quantization_annotation"
            ]._reuse_input_obs_or_fq
        else:
            if "target_dtype_info" not in node.meta:
                raise AssertionError("expected 'target_dtype_info' in node.meta")
            # TODO: we are assuming "target_dtype_info" exists here, maybe
            # a default value also need to be provided here
            target_dtype_info = node.meta["target_dtype_info"]
            # for nodes that doesn't have `reuse_input_obs_or_fq` configured,
            # we'll default to False, this makes configuring this field optional for users
            reuse_input_obs_or_fq = target_dtype_info.get(
                "reuse_input_obs_or_fq", False
            )
        arg_as_input_act_obs_or_fq = _get_arg_as_input_act_obs_or_fq(
            arg, node, named_modules, obs_or_fq_map, is_qat
        )
        (
            arg_as_input_target_dtype,
            arg_as_input_target_is_dynamic,
        ) = _get_dtype_and_is_dynamic(arg_as_input_act_obs_or_fq)

        arg_as_output_act_obs_or_fq = _get_output_act_obs_or_fq(
            arg, named_modules, obs_or_fq_map, is_qat
        )
        (
            arg_as_output_target_dtype,
            arg_as_output_target_is_dynamic,
        ) = _get_dtype_and_is_dynamic(arg_as_output_act_obs_or_fq)

        needs_obs_or_fq = _needs_obs_or_fq(
            arg_as_output_target_dtype,
            arg_as_output_target_is_dynamic,
            arg_as_input_target_dtype,
            arg_as_input_target_is_dynamic,
            reuse_input_obs_or_fq,
            is_zeroth_arg=len(node.args) > 0 and arg is node.args[0],
        )

    else:
        if qconfig is None:
            raise AssertionError("qconfig must not be None")
        # custom flow for standalone modules
        _, _, sm_prepare_custom_config, _ = _get_standalone_module_configs(
            node, named_modules, prepare_custom_config, qconfig, backend_config
        )
        sm_input_quantized_idxs = sm_prepare_custom_config.input_quantized_indexes

        # for args, this is set to the index of the current arg
        # for kwargs, this is left at None
        cur_input_idx = None
        for arg_idx, arg_to_check in enumerate(node.args):
            if arg_to_check is arg:
                cur_input_idx = arg_idx
                break

        if cur_input_idx is None:
            needs_obs_or_fq = False
        else:
            arg_as_output_target_dtype = _get_arg_target_dtype_as_output(
                arg, named_modules, obs_or_fq_map, is_qat
            )
            arg_as_input_target_dtype = (
                torch.quint8
                if cur_input_idx in sm_input_quantized_idxs
                else torch.float
            )
            needs_obs_or_fq = (
                arg_as_output_target_dtype != arg_as_input_target_dtype
            ) and (arg_as_input_target_dtype != torch.float)

        act_post_process_ctr = qconfig.activation
        arg_as_input_act_obs_or_fq = (
            act_post_process_ctr() if act_post_process_ctr else None
        )

    if needs_obs_or_fq:
        existing_obs_node = None

        # Before using the new observer, check if an observer
        # of the correct type already exists. If it does, use it.
        # This prevents duplicate observer insertions if a node is
        # used by multiple nodes.
        # TODO: this is looking into how the value is used in the future
        # we should remove this
        # removing this means we insert one observer for each use, even if they
        # have the same dtype, we can have an extra pass that removes the extra observers
        for maybe_obs_node in arg.users:
            if maybe_obs_node.op == "call_module":
                maybe_obs_mod = named_modules[maybe_obs_node.target]  # type: ignore[index]
                if (
                    type(maybe_obs_mod) is type(arg_as_input_act_obs_or_fq)
                    and maybe_obs_mod.dtype == arg_as_input_target_dtype  # type: ignore[possibly-undefined]
                ):
                    arg_as_input_act_obs_or_fq = maybe_obs_mod  # type: ignore[assignment]
                    existing_obs_node = maybe_obs_node
                    break

        if arg_as_input_act_obs_or_fq is None:
            raise AssertionError("arg_as_input_act_obs_or_fq must not be None")
        obs_or_fq_map[(arg, node)] = arg_as_input_act_obs_or_fq
        if existing_obs_node is None:
            new_obs_node = _insert_obs_or_fq(
                arg,
                arg_as_input_act_obs_or_fq,
                model,
                named_modules,
                graph,
                model_device,
            )
            # override this arg to be the observed arg
            new_arg = new_obs_node
        else:
            new_arg = existing_obs_node

    return new_arg


def _maybe_insert_input_observers_for_node(
    node: Node,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    qhandler: QuantizeHandler | None,
    prepare_custom_config: PrepareCustomConfig,
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
    backend_config: BackendConfig | None = None,
    model_device: torch.device | None = None,
) -> None:
    """
    If needed, inserts observers to the input args and kwargs of `node`.
    Note: modifies `node` inplace.

    For example, if cur_node needs an observer after prev_node, we change from

      prev_node -> cur_node

    To

      prev_node -> obs -> cur_node

    Note: backend_config only needed for standalone_module node
    """
    # Look through every input arg.  If that arg's target dtype does not
    # match the current node's target dtype, insert an observer.
    new_args = []
    for arg in node.args:
        new_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node,
            arg,
            qconfig,
            model,
            named_modules,
            graph,
            qhandler,
            prepare_custom_config,
            obs_or_fq_map,
            is_qat,
            backend_config,
            model_device,
        )
        new_args.append(new_arg)

    new_kwargs = {}
    for k, kwarg in node.kwargs.items():
        new_kwarg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node,
            kwarg,
            qconfig,
            model,
            named_modules,
            graph,
            qhandler,
            prepare_custom_config,
            obs_or_fq_map,
            is_qat,
            backend_config,
            model_device,
        )
        new_kwargs[k] = new_kwarg

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)
    node.kwargs = new_kwargs


def _maybe_insert_input_equalization_observers_for_node(
    node: Node,
    equalization_qconfig: Any,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    is_branch: bool,
) -> None:
    """
    If `node` needs to be equalized, find the input/weight observers it needs in
    `equalization_qconfig`, creates them, and inserts it into `graph`.

    If `node` does not need an equalization observer, returns None.
    """
    if equalization_qconfig is None or not node_supports_equalization(
        node, named_modules
    ):
        return

    if is_branch:
        warnings.warn(
            f"Cannot equalize {node} because it is part of a branch.", stacklevel=2
        )
        return

    new_args = []
    for arg in node.args:
        if not isinstance(arg, Node) or node_arg_is_bias(node, arg):
            new_args.append(arg)
            continue

        is_weight = node_arg_is_weight(node, arg)

        act_eq_process_ctr = (
            equalization_qconfig.weight
            if is_weight
            else equalization_qconfig.input_activation
        )

        new_eq_obs_mod = act_eq_process_ctr()
        new_eq_obs_node = _insert_obs_or_fq(
            arg, new_eq_obs_mod, model, named_modules, graph
        )

        new_args.append(new_eq_obs_node)

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)


def _maybe_insert_output_observer_for_node(
    node: Node,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> Node | None:
    """
    If `node` needs an output observer, creates it, inserts it into `graph`
    and returns it.

    If `node` does not need an output observer, returns None.

    Note: inserting dynamic quantization ops for output is not supported in fx graph mode
    quantization code path right now
    """
    if node.op == "output":
        raise AssertionError("observer insertion for outputs is handled elsewhere")

    is_standalone_module = False
    if "quantization_annotation" in node.meta:
        output_act_obs_or_fq = _create_obs_or_fq_from_qspec(
            node.meta["quantization_annotation"].output_qspec, obs_or_fq_map, is_qat
        )
    else:
        if "target_dtype_info" not in node.meta:
            raise AssertionError("expected 'target_dtype_info' in node.meta")
        is_standalone_module = node.meta["target_dtype_info"].get(
            "_is_standalone_module", False
        )
        output_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "output_act_obs_or_fq_ctr"
        )
        output_act_obs_or_fq = (
            output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
        )
    target_dtype, target_is_dynamic = _get_dtype_and_is_dynamic(output_act_obs_or_fq)
    # uncomment after we support reuse_input_obs_or_fq properly by having separate
    # implementations for this key instead of reusing the input_output_share_observers
    # code
    # reuse_input_obs_or_fq = node.meta["target_dtype_info"].get("reuse_input_obs_or_fq", False)
    # for now we set this to False since reuse_input_obs_or_fq for
    # the output of a node is implementation in the same code path as observer sharing,
    # we should refactor this part to make it clearer in the future
    # and we would be able to read this from config directly
    reuse_input_obs_or_fq = False

    # Note: prev_output_dtype = torch.float and prev_output_is_dynamic=False
    # because the prev_output is the output of an fp32 op, although technically
    # we should get the dtype of the output from node.meta["val"] in the future
    # if we deprecate fx graph mode quantization
    needs_obs_or_fq = _needs_obs_or_fq(
        torch.float, False, target_dtype, target_is_dynamic, reuse_input_obs_or_fq
    )
    # currently the activation in QConfig(activation=...,) is for both input
    # and output, and when the activation is configured to be dynamic quantization
    # e.g. PlaceholderObserver(dtype=torch.quint8, is_dynamic=True, ...), it means
    # the input should by dynamically quantized, but output should not be quantized
    #
    # there is no way we can specify different observer/fq for input and output
    # activation through QConfig today, this limitation is lifted in the
    # quantizer/annotation API in pytorch 2.0 export quantization code path,
    # but since this code is reused, annotating output to be dynamically quantized
    # would not work either for that.
    # we can change QConfig to support input/output activation if we want
    # to remove the following check, or if we can deprecate fx graph mode quantization
    if target_is_dynamic:
        needs_obs_or_fq = False

    # we never insert observers to output of standalone module, we assume
    # if needed, they are inserted inside the standalone module
    needs_obs_or_fq = needs_obs_or_fq and (not is_standalone_module)

    if needs_obs_or_fq:
        obs_or_fq_map[node] = output_act_obs_or_fq
        return _insert_obs_or_fq(
            node, output_act_obs_or_fq, model, named_modules, graph
        )
    else:
        return None


def _maybe_insert_observers_before_graph_output(
    graph_output_node: Node,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> None:
    """
    If the output needs to be quantized and there are any nodes
    in the output which are not already observed, inserts observers
    for those nodes.
    """

    def _recursive_maybe_replace_node_with_obs(
        maybe_node: Argument,
        model: torch.nn.Module,
        named_modules: dict[str, torch.nn.Module],
        graph: Graph,
    ) -> Argument:
        """
        Navigate an arbitrary data structure of lists, tuples, dicts.
        For each container type, recurse on all inputs. Once any Node
        is found, insert an observer if needed and do not recurse further.

        For example, given a structure of

          {'foo1': [[bar1]], 'foo2': {'foo3': [[[bar3]]]}}

        we recurse down to bar1 and bar3, observe them if necessary,
        and if we inserted an observer then replace the original node
        with its observer.

        Returns the data structure with all nodes needing observation being
        replaced by their observers.
        """
        if isinstance(maybe_node, Node):
            # check dtype of this node
            arg_as_output_target_dtype = _get_arg_target_dtype_as_output(
                maybe_node, named_modules, obs_or_fq_map, is_qat
            )
            observer_mod = None
            arg_as_input_target_dtype = torch.float
            if "target_dtype_info" in maybe_node.meta:
                observer_cls = maybe_node.meta["target_dtype_info"].get(
                    "input_act_obs_or_fq_ctr", None
                )
                if observer_cls is not None:
                    observer_mod = observer_cls()
                    arg_as_input_target_dtype = observer_mod.dtype
            # TODO: this does not handle dynamic quantization yet
            need_obs = (
                arg_as_output_target_dtype != arg_as_input_target_dtype
                and arg_as_input_target_dtype != torch.float
            )
            if need_obs:
                if observer_mod is None:
                    raise AssertionError(
                        "observer_mod must not be None when need_obs is True"
                    )
                # insert observer
                observer_node = _insert_obs_or_fq(
                    maybe_node, observer_mod, model, named_modules, graph
                )
                return observer_node
            else:
                return maybe_node
        elif isinstance(maybe_node, (list, tuple)):
            results = [
                _recursive_maybe_replace_node_with_obs(
                    inner_node, model, named_modules, graph
                )
                for inner_node in maybe_node
            ]
            if isinstance(maybe_node, list):
                return results
            else:
                return tuple(results)
        elif isinstance(maybe_node, dict):
            results_dict = {}
            for k, inner_v in maybe_node.items():
                results_dict[k] = _recursive_maybe_replace_node_with_obs(
                    inner_v, model, named_modules, graph
                )
            return results_dict
        elif maybe_node is None:
            return None
        else:
            raise Exception(  # noqa: TRY002
                "Unhandled type for returned node:", maybe_node
            )

    new_args = [
        _recursive_maybe_replace_node_with_obs(old_arg, model, named_modules, graph)
        for old_arg in graph_output_node.args
    ]

    graph_output_node.args = tuple(new_args)  # type: ignore[assignment]


def _maybe_propagate_dtype_for_node(
    node: Node,
    target_dtype: torch.dtype | type,
    node_name_to_match_result_with_qconfig: dict[str, _MatchResultWithQConfig],
) -> None:
    """
    Assigns `target_dtype` to `node`, setting `is_dynamic` to False. If `node`
    is a general tensor shape op, also call this function recursively on
    the first argument, to propagate the dtype to the caller.
    """
    node.meta["target_dtype_info"]["input_act_obs_or_fq_ctr"] = None
    node.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"] = None
   
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/quantization/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/ao/quantization/fx`):

- [`fuse_handler.py_docs.md_docs.md`](./fuse_handler.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`quantize_handler.py_kw.md_docs.md`](./quantize_handler.py_kw.md_docs.md)
- [`lstm_utils.py_kw.md_docs.md`](./lstm_utils.py_kw.md_docs.md)
- [`prepare.py_kw.md_docs.md`](./prepare.py_kw.md_docs.md)
- [`graph_module.py_docs.md_docs.md`](./graph_module.py_docs.md_docs.md)
- [`fuse_handler.py_kw.md_docs.md`](./fuse_handler.py_kw.md_docs.md)
- [`quantize_handler.py_docs.md_docs.md`](./quantize_handler.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`lower_to_qnnpack.py_kw.md_docs.md`](./lower_to_qnnpack.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `prepare.py_docs.md_docs.md`
- **Keyword Index**: `prepare.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
