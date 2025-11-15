# Documentation: `docs/torch/_functorch/_aot_autograd/frontend_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_functorch/_aot_autograd/frontend_utils.py_docs.md`
- **Size**: 16,985 bytes (16.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_functorch/_aot_autograd/frontend_utils.py`

## File Metadata

- **Path**: `torch/_functorch/_aot_autograd/frontend_utils.py`
- **Size**: 13,406 bytes (13.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

import warnings
from collections.abc import KeysView
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._guards import detect_fake_mode
from torch._library.opaque_object import is_opaque_type
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import _pytree_subclasses_that_lose_info
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config
from .descriptors import BufferAOTInput, DifferentiableAOTInput, ParamAOTInput
from .schemas import AOTConfig, FakifiedFlatArgs


static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)


def process_inputs(
    flat_args: list[Any],
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: Optional[ShapeEnv],
    ignore_shape_env: bool = False,
) -> FakifiedFlatArgs:
    with fake_mode:

        def convert(idx, x):
            if shape_env is not None and not ignore_shape_env:
                from torch._dynamo.source import ConstantSource

                if isinstance(x, int):
                    # We always specialize on scalar values in export.
                    if aot_config.is_export:
                        return x
                    source = ConstantSource(f"sym_{idx}")
                    return shape_env.create_symintnode(
                        shape_env.create_symbol(x, source, positive=x >= 0),
                        hint=x,
                        source=source,
                    )
            if isinstance(x, torch.ScriptObject) or is_opaque_type(type(x)):
                return torch._library.fake_class_registry.maybe_to_fake_obj(
                    fake_mode, x
                )
            if not isinstance(x, torch.Tensor):
                return x
            if isinstance(x, FakeTensor):
                assert x.fake_mode is fake_mode
                return x
            if is_traceable_wrapper_subclass(x):
                attrs, _ = x.__tensor_flatten__()
                if all(isinstance(getattr(x, attr), FakeTensor) for attr in attrs):
                    assert all(
                        getattr(x, attr).fake_mode is fake_mode for attr in attrs
                    )
                    return x

            # see note [Tensor Fakification and Symbol Caching]
            symbolic_context = None
            source = None
            trace = True
            if tracing_context := torch._guards.TracingContext.try_get():
                if x in tracing_context.tensor_to_context:
                    symbolic_context = tracing_context.tensor_to_context[x]
                    source = symbolic_context.tensor_source
                    # We already fakeified this tensor in Dynamo, don't
                    # dump the trace for it again
                    trace = False
            if (
                idx < aot_config.num_params_buffers
                and config.static_weight_shapes
                and not symbolic_context
            ):
                # TODO: Ensure that this codepath is never exercised from
                # Dynamo
                return fake_mode.from_tensor(x, static_shapes=True)

            result = fake_mode.from_tensor(
                x,
                static_shapes=ignore_shape_env,
                symbolic_context=symbolic_context,
                source=source,
                trace=trace,
            )
            return result

        return FakifiedFlatArgs([convert(idx, x) for idx, x in enumerate(flat_args)])


def construct_fake_mode(
    flat_args: list[Any], aot_config: AOTConfig
) -> tuple[FakeTensorMode, Optional[ShapeEnv]]:
    fake_mode = detect_fake_mode(flat_args)
    if fake_mode is None:
        shape_env = ShapeEnv() if aot_config.dynamic_shapes else None
        fake_mode = FakeTensorMode(shape_env=shape_env)
    else:
        shape_env = fake_mode.shape_env
    return (fake_mode, shape_env)


def _try_get_metadata_from_dynamo(
    mod: torch.nn.Module,
    param_keys: KeysView[str],
    full_args_num: int,
    full_args_descs: list[DifferentiableAOTInput],
) -> tuple[Optional[list[torch._guards.Source]], list[int]]:
    """
    Metadata is forwarded from Dynamo to AOTDispatch via special fields on GraphModule.
    We first verify that `mod` does come from Dynamo, then we handle cases where
    metadata might be missing.

    Returns:
        aot_autograd_arg_pos_to_source: used to dedup params and their guards
        static_input_indices: used to identify static inputs for cudagraphs
    """
    # Note [Assumption on Dynamo Metadata]
    # This function assumes a graph module from dynamo provides `dynamo_compiled_id`,
    # _param_name_to_source, and every placeholder node has `_dynamo_source` attributes.
    # When gm is modified (e.g., DDPOptimizer via split_module), metadata needs to
    # be propagated in order to be recognized as a dynamo graph

    if not (isinstance(mod, torch.fx.GraphModule) and "dynamo_compile_id" in mod.meta):
        # graph was not captured by dynamo
        return None, []

    if not hasattr(mod, "_param_name_to_source"):
        # is from export
        static_input_indices = [
            i
            for i, node in enumerate(full_args_descs)
            if isinstance(node, (ParamAOTInput, BufferAOTInput))
        ]
        return None, static_input_indices

    # We now know this came from dynamo, and (1) we care about guards,
    # so setting up aot_autograd_arg_pos_to_source for downstream dedup guards
    # can now be done safely. (2) Dynamo logic protects the 1:1 sizing below.
    # Additionally, we mark static indices for cudagraphs.
    param_name_to_source = mod._param_name_to_source
    seen_sources = set()

    aot_autograd_arg_pos_to_source = []
    static_input_indices = []
    # Collect the new inputs lifted by aotdispatch
    for i, name in enumerate(param_keys):
        assert name in param_name_to_source, f"{name} not found."
        source = param_name_to_source[name]
        assert source not in seen_sources, source
        seen_sources.add(source)
        aot_autograd_arg_pos_to_source.append(source)

        static_input_indices.append(i)

    # Collect the dynamo graph inputs
    # TODO(mlazos): Revisit if this is still needed. With Dynamo install ID
    # matched tensors back into the Fx graph, this might not be necessary.
    for pos, node in enumerate(mod.graph.find_nodes(op="placeholder")):
        assert hasattr(node, "_dynamo_source")
        source = node._dynamo_source
        # `source`` specifies the source from user code. ddp optimizer may have
        # intermediate values becoming submodule placeholders which does not
        # have a source
        assert source is None or source not in seen_sources, source
        seen_sources.add(source)
        aot_autograd_arg_pos_to_source.append(source)
        source_name = source.name() if source else str(source)

        # input[i] in dynamo is now:
        # input[i + len(extra_params)] in AOT,
        # where extra_params are the params/buffers that dynamo baked into the
        # OutputGraph
        actual_pos = pos + len(param_keys)

        if "tensor_dict" in node.meta and node.meta["tensor_dict"].get(
            "_dynamo_static_input_type", None
        ):
            static_inputs_log.debug(
                "Adding static input pos %s for source %s", actual_pos, source_name
            )
            static_input_indices.append(actual_pos)
        else:
            static_inputs_log.debug(
                "Non-static input pos %s for source %s", actual_pos, source_name
            )

    assert full_args_num == len(aot_autograd_arg_pos_to_source)
    return aot_autograd_arg_pos_to_source, static_input_indices


@contextmanager
def _detect_attribute_assignment(mod: torch.nn.Module):
    # Do not allow assignment of tensor attributes during export unless
    # the attribute is registered as a buffer.

    NN_MODULE_STD_ATTRS = [
        "_backward_hooks",
        "_backward_pre_hooks",
        "_buffers",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_is_full_backward_hook",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_modules",
        "_non_persistent_buffers_set",
        "_parameters",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "training",
    ]
    NN_MODULE_LAZY_STD_ATTRS = [
        "_initialize_hook",
        "_load_hook",
    ]
    STD_ATTRS = {
        *NN_MODULE_STD_ATTRS,
        *NN_MODULE_LAZY_STD_ATTRS,
    }

    def _get_attributes(mod):
        # return any attributes of a module that are not standard attributes
        return {k: v for k, v in mod.__dict__.items() if k not in STD_ATTRS}

    def _get_all_module_attributes(mod):
        # return attributes from all modules and submodules
        result = {}
        for name, submodule in mod.named_modules():
            result[name] = _get_attributes(submodule)
        return result

    def _restore_all_module_attributes(mod, snapshot):
        # restore attributes to all modules and submodules
        for name, submodule in mod.named_modules():
            if name in snapshot:
                submodule.__dict__.update(snapshot[name])

    # save state of attributes before enter
    snapshot = pytree.tree_map(
        lambda x: x,
        _get_all_module_attributes(mod),
        is_leaf=lambda x: type(x) in _pytree_subclasses_that_lose_info,
    )
    try:
        yield
    finally:
        # after exit, compare state of attributes with snapshot
        # to detect which tensor attributes were assigned

        def _collect_assigned_tensor_attributes(snapshot, new_attrs):
            assigned_tensor_attributes = []

            def _compare_values(path, old_val, new_val):
                """Recursively compare values, handling containers."""
                # Same object, no change
                if old_val is new_val:
                    return

                if old_val is None or new_val is None:
                    if isinstance(new_val, torch.Tensor):
                        assigned_tensor_attributes.append(path)
                    return

                # Check if it's a tensor that was reassigned
                if isinstance(new_val, torch.Tensor):
                    assigned_tensor_attributes.append(path)
                    return

                # Handle dict containers
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    all_keys = set(old_val.keys()) | set(new_val.keys())
                    for key in all_keys:
                        old_item = old_val.get(key)
                        new_item = new_val.get(key)
                        _compare_values(f"{path}[{key!r}]", old_item, new_item)
                    return

                # Handle list/tuple containers
                if isinstance(old_val, (list, tuple)) and isinstance(
                    new_val, (list, tuple)
                ):
                    # Different lengths = mutation happened
                    max_len = max(len(old_val), len(new_val))
                    for i in range(max_len):
                        old_item = old_val[i] if i < len(old_val) else None
                        new_item = new_val[i] if i < len(new_val) else None
                        _compare_values(f"{path}[{i}]", old_item, new_item)
                    return

                # For other types, just check if they're different objects
                # (we don't care about non-tensor mutations)

            for module_name in snapshot.keys() | new_attrs.keys():
                old_module_attrs = snapshot.get(module_name, {})
                new_module_attrs = new_attrs.get(module_name, {})

                for attr_name in old_module_attrs.keys() | new_module_attrs.keys():
                    module_prefix = f"self.{module_name}." if module_name else "self."
                    full_path = f"{module_prefix}{attr_name}"

                    old_val = old_module_attrs.get(attr_name)
                    new_val = new_module_attrs.get(attr_name)
                    _compare_values(full_path, old_val, new_val)

            return assigned_tensor_attributes

        new_attrs = _get_all_module_attributes(mod)
        assigned_tensor_attributes = _collect_assigned_tensor_attributes(
            snapshot, new_attrs
        )
        # restore state of all attributes (including, e.g., of primitive types)
        _restore_all_module_attributes(mod, snapshot)

        if assigned_tensor_attributes:
            if len(assigned_tensor_attributes) > 1:
                noun, verb = "attributes", "were"
            else:
                noun, verb = "attribute", "was"
            warnings.warn(
                f"The tensor {noun} {', '.join(assigned_tensor_attributes)} {verb} assigned during export. "
                "Such attributes must be registered as buffers using the `register_buffer` API "
                "(https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer).",
                stacklevel=2,
            )

```



## High-Level Overview


This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `process_inputs`, `convert`, `construct_fake_mode`, `_try_get_metadata_from_dynamo`, `_detect_attribute_assignment`, `_get_attributes`, `_get_all_module_attributes`, `_restore_all_module_attributes`, `_collect_assigned_tensor_attributes`, `_compare_values`

**Key imports**: warnings, KeysView, contextmanager, Any, Optional, torch, torch.utils._pytree as pytree, detect_fake_mode, is_opaque_type, FakeTensor, FakeTensorMode, _pytree_subclasses_that_lose_info


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_functorch/_aot_autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `warnings`
- `collections.abc`: KeysView
- `contextlib`: contextmanager
- `typing`: Any, Optional
- `torch`
- `torch.utils._pytree as pytree`
- `torch._guards`: detect_fake_mode
- `torch._library.opaque_object`: is_opaque_type
- `torch._subclasses`: FakeTensor, FakeTensorMode
- `torch.fx.experimental.proxy_tensor`: _pytree_subclasses_that_lose_info
- `torch.fx.experimental.symbolic_shapes`: ShapeEnv
- `torch.utils._python_dispatch`: is_traceable_wrapper_subclass
- `..`: config
- `.descriptors`: BufferAOTInput, DifferentiableAOTInput, ParamAOTInput
- `.schemas`: AOTConfig, FakifiedFlatArgs
- `torch._dynamo.source`: ConstantSource


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/_functorch/_aot_autograd`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_cache.py_docs.md`](./autograd_cache.py_docs.md)
- [`functional_utils.py_docs.md`](./functional_utils.py_docs.md)
- [`logging_utils.py_docs.md`](./logging_utils.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`descriptors.py_docs.md`](./descriptors.py_docs.md)
- [`collect_metadata_analysis.py_docs.md`](./collect_metadata_analysis.py_docs.md)
- [`subclass_parametrization.py_docs.md`](./subclass_parametrization.py_docs.md)
- [`runtime_wrappers.py_docs.md`](./runtime_wrappers.py_docs.md)


## Cross-References

- **File Documentation**: `frontend_utils.py_docs.md`
- **Keyword Index**: `frontend_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_functorch/_aot_autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_functorch/_aot_autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/_functorch/_aot_autograd`):

- [`graph_compile.py_kw.md_docs.md`](./graph_compile.py_kw.md_docs.md)
- [`frontend_utils.py_kw.md_docs.md`](./frontend_utils.py_kw.md_docs.md)
- [`autograd_cache.py_docs.md_docs.md`](./autograd_cache.py_docs.md_docs.md)
- [`input_output_analysis.py_kw.md_docs.md`](./input_output_analysis.py_kw.md_docs.md)
- [`schemas.py_docs.md_docs.md`](./schemas.py_docs.md_docs.md)
- [`collect_metadata_analysis.py_docs.md_docs.md`](./collect_metadata_analysis.py_docs.md_docs.md)
- [`functional_utils.py_docs.md_docs.md`](./functional_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`logging_utils.py_docs.md_docs.md`](./logging_utils.py_docs.md_docs.md)
- [`graph_capture.py_kw.md_docs.md`](./graph_capture.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `frontend_utils.py_docs.md_docs.md`
- **Keyword Index**: `frontend_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
