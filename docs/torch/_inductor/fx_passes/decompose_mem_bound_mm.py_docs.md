# Documentation: `torch/_inductor/fx_passes/decompose_mem_bound_mm.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/decompose_mem_bound_mm.py`
- **Size**: 11,217 bytes (10.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import logging

import torch
from torch import Tensor
from torch._dynamo.utils import counters, is_node_meta_valid
from torch.fx.experimental.symbolic_shapes import (
    statically_known_false,
    statically_known_true,
)

from .. import config
from ..pattern_matcher import Arg, CallFunction, Match, register_graph_pattern
from .split_cat import construct_pattern_matcher_pass


aten = torch.ops.aten
log = logging.getLogger(__name__)

# TODO: need a better strategy for decomposing mm
# The following two constants are for CUDA device only
MIN_FIRST_DIMENSION_DECOMPOSITION = 10240
MAX_OTHER_DIMENSION_DECOMPOSITION = 32
# The following two constants are for CPU device only
CPU_MAX_FIRST_DIMENSION_DECOMPOSITION = 1
CPU_MAX_OTHER_DIMENSION_DECOMPOSITION = 2048

min_first_dimension_decomposition = MIN_FIRST_DIMENSION_DECOMPOSITION
max_other_dimension_decomposition = MAX_OTHER_DIMENSION_DECOMPOSITION
cpu_max_first_dimension_decomposition = CPU_MAX_FIRST_DIMENSION_DECOMPOSITION
cpu_max_other_dimension_decomposition = CPU_MAX_OTHER_DIMENSION_DECOMPOSITION
if "decompose_mm_pass" in config.post_grad_fusion_options:
    min_first_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("min_first_dimension_decomposition", MIN_FIRST_DIMENSION_DECOMPOSITION)
    max_other_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("max_other_dimension_decomposition", MAX_OTHER_DIMENSION_DECOMPOSITION)
    cpu_max_first_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get(
        "cpu_max_first_dimension_decomposition", CPU_MAX_FIRST_DIMENSION_DECOMPOSITION
    )
    cpu_max_other_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get(
        "cpu_max_other_dimension_decomposition", CPU_MAX_OTHER_DIMENSION_DECOMPOSITION
    )


def check_device(a: Tensor, b: Tensor, device="cuda") -> bool:
    return (a.device.type == b.device.type) and (b.device.type == device)


def realize_inputs(inputs: list[torch.fx.Node]):
    for inp in inputs:
        if isinstance(inp, torch.fx.node.Node):
            inp.meta["inductor_realize_to_strides"] = True


def should_decompose_bmm(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    if len(mat1.shape) != 3 or len(mat2.shape) != 3:
        return False
    if check_device(mat1, mat2, device="cuda") or check_device(
        mat1, mat2, device="xpu"
    ):
        if mat1.shape[0] < min_first_dimension_decomposition:
            return False
        # 2 of m, n, k must be <= MAX_OTHER_DIMENSION_DECOMPOSITION
        # use bool() to deal with BooleanAtom type
        if (
            bool(mat1.shape[1] < max_other_dimension_decomposition)
            + bool(mat1.shape[2] < max_other_dimension_decomposition)
            + bool(mat2.shape[2] < max_other_dimension_decomposition)
            < 2
        ):
            return False
        return True
    elif check_device(mat1, mat2, device="cpu"):
        if (
            mat1.shape[0] <= cpu_max_first_dimension_decomposition
            and mat2.shape[0] <= cpu_max_first_dimension_decomposition
        ):
            return True
    return False


def should_decompose_mm(mat1, mat2) -> bool:
    """
    Determines whether matrix multiplication (mm) should be decomposed into pointwise operations
    based on the input matrices' metadata, shapes, device placement, and configuration options.
    Args:
        mat1: The first matrix operand. Expected to be an object with a `.meta` attribute containing
              a "val" key, or a tensor-like object with a `.shape` attribute.
        mat2: The second matrix operand. Same requirements as `mat1`.
    Returns:
        bool: True if the matrix multiplication should be decomposed according to the following logic:
            - Both inputs must have valid node metadata.
            - Both matrices must be 2-dimensional.
            - If the configuration option `skip_dynamic_shape_dim_check` is False:
                - Decomposition is only considered for statically-shaped matrices.
                - For CUDA devices: `mat1.shape[0]` must be at least `min_first_dimension_decomposition`,
                  and both dimensions of `mat2` must be less than `max_other_dimension_decomposition`.
                - For CPU devices: All relevant dimensions must be less than or equal to their respective
                  CPU decomposition thresholds.
            - If `skip_dynamic_shape_dim_check` is True:
                - Decomposition is considered for dynamic shapes as well, using a combination of
                  `statically_known_true` and `statically_known_false` checks to handle uncertainty.
                - The same dimension and device checks apply, but allow for dynamic/static uncertainty.
            - Returns False if any of the above conditions are not met.
    Notes:
        - Relies on helper functions such as `is_node_meta_valid`, `check_device`, `statically_known_true`,
          and `statically_known_false`, as well as configuration values like
          `min_first_dimension_decomposition`, `max_other_dimension_decomposition`, etc.
        - Designed for use in graph optimization or fusion passes where decomposing large or dynamic
          matrix multiplications can improve performance or memory usage.
    """
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    if len(mat1.shape) != 2 or len(mat2.shape) != 2:
        return False
    # case 1: we skip decompose mm if the input is dynamic shape
    if not config.post_grad_fusion_options["decompose_mm_pass"].get(
        "skip_dynamic_shape_dim_check", False
    ):
        return (
            (
                check_device(mat1, mat2, device="cuda")
                or check_device(mat1, mat2, device="xpu")
            )
            and statically_known_true(
                mat1.shape[0] >= min_first_dimension_decomposition
            )
            and statically_known_true(mat2.shape[0] < max_other_dimension_decomposition)
            and statically_known_true(mat2.shape[1] < max_other_dimension_decomposition)
        ) or (
            check_device(mat1, mat2, device="cpu")
            and statically_known_true(
                mat1.shape[0] <= cpu_max_first_dimension_decomposition
            )
            and statically_known_true(
                mat2.shape[0] <= cpu_max_other_dimension_decomposition
            )
            and statically_known_true(
                mat2.shape[1] <= cpu_max_other_dimension_decomposition
            )
        )
    # case 2: we decompose mm if the input is dynamic shape
    else:
        return (
            (
                check_device(mat1, mat2, device="cuda")
                or check_device(mat1, mat2, device="xpu")
            )
            and (
                statically_known_true(
                    mat1.shape[0] >= min_first_dimension_decomposition
                )
                or not statically_known_false(
                    mat1.shape[0] >= min_first_dimension_decomposition
                )
            )
            and (
                statically_known_true(mat2.shape[0] < max_other_dimension_decomposition)
                or not statically_known_false(
                    mat2.shape[0] < max_other_dimension_decomposition
                )
            )
            and (
                statically_known_true(mat2.shape[1] < max_other_dimension_decomposition)
                or not statically_known_false(
                    mat2.shape[1] < max_other_dimension_decomposition
                )
            )
        ) or (
            check_device(mat1, mat2, device="cpu")
            and (
                statically_known_true(
                    mat1.shape[0] <= cpu_max_first_dimension_decomposition
                )
                or not statically_known_false(
                    mat1.shape[0] <= cpu_max_first_dimension_decomposition
                )
            )
            and (
                statically_known_true(
                    mat2.shape[0] <= cpu_max_other_dimension_decomposition
                )
                or not statically_known_false(
                    mat2.shape[0] <= cpu_max_other_dimension_decomposition
                )
            )
            and (
                statically_known_true(
                    mat2.shape[1] <= cpu_max_other_dimension_decomposition
                )
                or not statically_known_false(
                    mat2.shape[1] <= cpu_max_other_dimension_decomposition
                )
            )
        )


def print_decompose_pattern(match: Match, inputs: list[torch.fx.Node]):
    node = match.nodes[-1]
    log.debug(
        "Decompose %s with input shape: %s",
        node.target,
        ", ".join(
            str(input.meta["val"].shape) if "val" in input.meta else "None"
            for input in inputs
        ),
    )


@register_graph_pattern(
    CallFunction(aten.bmm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_bmm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, :, None] * mat2[:, None, :, :], dim=-2).to(
            mat1.dtype
        )

    if should_decompose_bmm(mat1, mat2):
        counters["inductor"]["decompose_bmm"] += 1
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
        realize_inputs([mat1, mat2])
    return


@register_graph_pattern(
    CallFunction(aten.addmm, Arg(), Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_addmm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
    mat3: torch.fx.Node,
):
    def repl(mat1, mat2, mat3):
        return (
            torch.sum(mat2[:, :, None] * mat3[None, :, :], dim=-2).to(mat2.dtype) + mat1
        )

    if should_decompose_mm(mat2, mat3):
        counters["inductor"]["decompose_addmm"] += 1
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2, mat3])
        print_decompose_pattern(match, [mat1, mat2, mat3])
        realize_inputs([mat1, mat2, mat3])
    return


@register_graph_pattern(
    CallFunction(aten.mm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_mm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, None] * mat2[None, :, :], dim=-2).to(mat1.dtype)

    if should_decompose_mm(mat1, mat2):
        counters["inductor"]["decompose_mm"] += 1
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
        realize_inputs([mat1, mat2])
    return

```



## High-Level Overview


This Python file contains 0 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `check_device`, `realize_inputs`, `should_decompose_bmm`, `should_decompose_mm`, `print_decompose_pattern`, `decompose_bmm`, `repl`, `decompose_addmm`, `repl`, `decompose_mm`, `repl`

**Key imports**: logging, torch, Tensor, counters, is_node_meta_valid, config, Arg, CallFunction, Match, register_graph_pattern, construct_pattern_matcher_pass


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `torch`
- `torch._dynamo.utils`: counters, is_node_meta_valid
- `..`: config
- `..pattern_matcher`: Arg, CallFunction, Match, register_graph_pattern
- `.split_cat`: construct_pattern_matcher_pass


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

- **File Documentation**: `decompose_mem_bound_mm.py_docs.md`
- **Keyword Index**: `decompose_mem_bound_mm.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
