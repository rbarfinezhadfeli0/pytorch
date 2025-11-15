# Documentation: `torch/_inductor/fx_passes/freezing_patterns.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/freezing_patterns.py`
- **Size**: 9,641 bytes (9.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import functools

import torch
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.utils import GPU_TYPES

from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
    _return_true,
    CallFunction,
    fwd_only,
    Ignored,
    init_once_fakemode,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)


aten = torch.ops.aten

# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]

binary_folding_pass = PatternMatcherPass()


def freezing_passes(gm: torch.fx.GraphModule, aot_example_inputs):
    """
    Passes that are applied to the graph to freeze pass.
    """

    from ..freezing import constant_fold

    lazy_init()
    # We need a few rounds of binary folding to get rid of all the
    # unnecessary nodes, but may need a good method to chose the rounds number.
    # works like: conv+binary+binary.
    binary_folding = counters["inductor"]["binary_folding"]
    fake_tensor_prop(gm, aot_example_inputs, True)

    torch._inductor.fx_passes.binary_folding.mark_mixed_dtype_allowed_computation_ops(
        gm
    )
    for _ in range(4):
        constant_fold(gm)
        # Make sure meta['val'] is properly set for all nodes
        fake_tensor_prop(gm, aot_example_inputs, True)
        binary_folding_pass.apply(gm.graph)  # type: ignore[arg-type]
        # If we don't have binary folding, we don't need to run the pass again.
        # TODO: remove the need to run fake_tensor_prop on the whole model.
        if counters["inductor"]["binary_folding"] == binary_folding:
            break
        binary_folding = counters["inductor"]["binary_folding"]

    torch._inductor.fx_passes.binary_folding.recover_original_precision_folded_computation_ops(
        gm
    )

    constant_fold(gm)
    fake_tensor_prop(gm, aot_example_inputs, True)

    for pattern in pass_patterns:
        pattern.apply(gm.graph)  # type: ignore[arg-type]

    # The CPU weight packing always assume the conv's weight is channels last,
    # So make sure the layout_optimization is on when doing it.
    if (
        torch._C._has_mkldnn
        and config.cpp.weight_prepack
        and config.layout_optimization
    ):
        from .mkldnn_fusion import _eliminate_duplicate_packed_nodes

        _eliminate_duplicate_packed_nodes(gm)

    stable_topological_sort(gm.graph)
    gm.recompile()
    gm.graph.lint()


@init_once_fakemode
def lazy_init():
    if torch._C._has_mkldnn and config.cpp.weight_prepack:
        from .mkldnn_fusion import _mkldnn_weight_pack_init

        _mkldnn_weight_pack_init()

    from .binary_folding import binary_folding_init

    addmm_patterns_init()
    binary_folding_init()


def register_freezing_graph_pattern(pattern, extra_check=_return_true, pass_number=0):
    while pass_number > len(pass_patterns) - 1:
        pass_patterns.append(PatternMatcherPass())
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        # pyrefly: ignore [bad-argument-type]
        pass_dict=pass_patterns[pass_number],
    )


def register_binary_folding_pattern(pattern, extra_check=_return_true):
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        # pyrefly: ignore [bad-argument-type]
        pass_dict=binary_folding_pass,
    )


@functools.cache
def addmm_patterns_init():
    """
    addmm related patterns.
    To avoid duplication, also includes int8 WoQ GEMM pattern without bias.
    """
    device = next(
        (gpu for gpu in GPU_TYPES if getattr(torch, gpu).is_available()), "cpu"
    )
    val = functools.partial(torch.empty, (10, 10), device=device, requires_grad=False)
    scale = functools.partial(torch.empty, (10,), device=device, requires_grad=False)

    def check_int8_woq_concat_linear_weights(match):
        is_cpu = match.kwargs["inp"].meta["val"].is_cpu
        if not is_cpu or not config.cpp.enable_concat_linear:
            # Currently, this pattern is only supported on CPU
            return False

        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        if not all(
            match.kwargs[wgt].target is torch.ops.prims.convert_element_type.default
            for wgt in weight_inputs
        ):
            return False

        if not all(
            next(iter(match.kwargs[wgt]._input_nodes.keys())).meta["val"].dtype
            is torch.int8
            for wgt in weight_inputs
        ):
            return False

        if not all(
            match.kwargs[wgt].meta["val"].dtype is torch.bfloat16
            for wgt in weight_inputs
        ):
            return False

        return True

    def check_concat_weights(match):
        is_cpu = match.kwargs["inp"].meta["val"].is_cpu
        if is_cpu and not config.cpp.enable_concat_linear:
            return False

        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        equal_shape_inputs = [weight_inputs]

        if "b1" in match.kwargs:
            bias_inputs = ["b1", "b2"]
            if "b3" in match.kwargs:
                bias_inputs.append("b3")

            equal_shape_inputs.append(bias_inputs)

        for equal_shape_group in equal_shape_inputs:
            inps = [match.kwargs[name] for name in equal_shape_group]

            if not all(
                inp.op == "get_attr"
                and inp.meta["val"].shape == inps[0].meta["val"].shape
                for inp in inps
            ):
                return False
        return True

    def int8_woq_fusion_pattern(inp, w1, w2, w3, s1, s2, s3):
        return ((inp @ w1) * s1, (inp @ w2) * s2, (inp @ w3) * s3)

    def int8_woq_fusion_replacement(inp, w1, w2, w3, s1, s2, s3):
        cat_w = torch.cat((w1, w2, w3), dim=1)
        cat_s = torch.cat((s1, s2, s3), dim=0)
        mm = (inp @ cat_w).mul(cat_s)
        n1, n2 = w1.size(1), w2.size(1)
        return mm.tensor_split([n1, n1 + n2], dim=-1)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        int8_woq_fusion_pattern,
        # pyrefly: ignore [bad-argument-type]
        int8_woq_fusion_replacement,
        [val(), val(), val(), val(), scale(), scale(), scale()],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_int8_woq_concat_linear_weights,
        exclusive_arg_names=("w1", "w2", "w3", "s1", "s2", "s3"),
    )

    def matmul_fuse_pattern(inp, w1, w2, w3):
        return (inp @ w1, inp @ w2, inp @ w3)

    def matmul_replacement(inp, w1, w2, w3):
        cat_t = torch.cat((w1, w2, w3), dim=1)
        mm = inp @ cat_t
        return mm.chunk(3, dim=1)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        matmul_fuse_pattern,
        # pyrefly: ignore [bad-argument-type]
        matmul_replacement,
        [val(), val(), val(), val()],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3"),
    )

    def matmul_fuse_pattern_two(inp, w1, w2):
        return (inp @ w1, inp @ w2)

    def matmul_replacement_two(inp, w1, w2):
        cat_t = torch.cat((w1, w2), dim=1)
        mm = inp @ cat_t
        return mm.chunk(2, dim=1)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        matmul_fuse_pattern_two,
        # pyrefly: ignore [bad-argument-type]
        matmul_replacement_two,
        [val(), val(), val()],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2"),
    )

    def addmm_fuse_pattern_second(inp, w1, w2, w3, b1, b2, b3):
        return (
            aten.addmm(b1, inp, w1),
            aten.addmm(b2, inp, w2),
            aten.addmm(b3, inp, w3),
        )

    def addmm_fuse_replacement_second(inp, w1, w2, w3, b1, b2, b3):
        cat_w = torch.cat((w1, w2, w3), dim=1)
        cat_b = torch.cat((b1, b2, b3))
        return aten.addmm(cat_b, inp, cat_w).chunk(3, dim=1)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        addmm_fuse_pattern_second,
        # pyrefly: ignore [bad-argument-type]
        addmm_fuse_replacement_second,
        [val() for _ in range(7)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3", "b1", "b2", "b3"),
    )


def same_dtype(match):
    return match.output_node().args[0].meta["val"].dtype == match.kwargs["dtype"]


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        Ignored(),
        KeywordArg("dtype"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[0],
    extra_check=same_dtype,
)
def unnecessary_dtype_convert(match: Match, **kwargs):
    """Remove unnecessary dtype conversion op, probably left as a result of Conv-Bn folding"""
    graph = match.graph
    node = match.output_node()
    node.replace_all_uses_with(node.args[0])  # type: ignore[arg-type]
    graph.erase_node(node)

```



## High-Level Overview

"""    Passes that are applied to the graph to freeze pass.

This Python file contains 0 class(es) and 17 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `freezing_passes`, `lazy_init`, `register_freezing_graph_pattern`, `register_binary_folding_pattern`, `addmm_patterns_init`, `check_int8_woq_concat_linear_weights`, `check_concat_weights`, `int8_woq_fusion_pattern`, `int8_woq_fusion_replacement`, `matmul_fuse_pattern`, `matmul_replacement`, `matmul_fuse_pattern_two`, `matmul_replacement_two`, `addmm_fuse_pattern_second`, `addmm_fuse_replacement_second`, `same_dtype`, `unnecessary_dtype_convert`

**Key imports**: functools, torch, fake_tensor_prop, GPU_TYPES, counters, config, constant_fold, _eliminate_duplicate_packed_nodes, _mkldnn_weight_pack_init, binary_folding_init


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `torch`
- `torch._inductor.compile_fx`: fake_tensor_prop
- `torch._inductor.utils`: GPU_TYPES
- `..._dynamo.utils`: counters
- `..`: config
- `..freezing`: constant_fold
- `.mkldnn_fusion`: _eliminate_duplicate_packed_nodes
- `.binary_folding`: binary_folding_init


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/_inductor/fx_passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fuse_attention.py_docs.md`](./fuse_attention.py_docs.md)
- [`efficient_conv_bn_eval.py_docs.md`](./efficient_conv_bn_eval.py_docs.md)
- [`bucketing.py_docs.md`](./bucketing.py_docs.md)
- [`numeric_utils.py_docs.md`](./numeric_utils.py_docs.md)
- [`dedupe_symint_uses.py_docs.md`](./dedupe_symint_uses.py_docs.md)
- [`post_grad.py_docs.md`](./post_grad.py_docs.md)
- [`joint_graph.py_docs.md`](./joint_graph.py_docs.md)
- [`fsdp.py_docs.md`](./fsdp.py_docs.md)


## Cross-References

- **File Documentation**: `freezing_patterns.py_docs.md`
- **Keyword Index**: `freezing_patterns.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
