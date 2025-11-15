# Documentation: `docs/torch/_inductor/quantized_lowerings.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/quantized_lowerings.py_docs.md`
- **Size**: 8,757 bytes (8.55 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/quantized_lowerings.py`

## File Metadata

- **Path**: `torch/_inductor/quantized_lowerings.py`
- **Size**: 5,728 bytes (5.59 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import logging
from typing import Any

import torch
from torch._inductor.kernel.mm_common import mm_args

from . import config, lowering
from .codegen.cpp_gemm_template import CppGemmTemplate, CppWoqInt4GemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .lowering import expand, register_lowering
from .mkldnn_ir import WeightInt4PackMatmul
from .select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
)
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template
from .virtualized import V


log = logging.getLogger(__name__)

aten__weight_int8pack_mm = ExternKernelChoice(
    torch._weight_int8pack_mm, "at::_weight_int8pack_mm", has_out_variant=False
)

aten__weight_int4pack_mm_cpu = ExternKernelChoice(
    torch.ops.quantized.int4mm_packed_weight_cpu,
    "at::native::_weight_int4pack_mm_cpu_tensor",
    has_out_variant=False,
    kernel_creator=WeightInt4PackMatmul.create,
)

quantized = torch.ops.quantized
_quantized = torch.ops._quantized
aten = torch.ops.aten


def register_quantized_ops() -> None:
    lowering.add_needs_realized_inputs(
        [
            quantized.max_pool2d,
            _quantized.wrapped_fbgemm_pack_gemm_matrix_fp16,
            _quantized.wrapped_fbgemm_linear_fp16_weight,
        ]
    )
    lowering.make_fallback(quantized.max_pool2d)
    lowering.make_fallback(_quantized.wrapped_fbgemm_pack_gemm_matrix_fp16)
    lowering.make_fallback(_quantized.wrapped_fbgemm_linear_fp16_weight)


def register_woq_mm_ops() -> None:
    @register_lowering(aten._weight_int8pack_mm, type_promotion_kind=None)  # type: ignore[misc]
    def int8pack_mm(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        layout: Any = None,
    ) -> Any:
        _, _, _, layout, mat1, mat2 = mm_args(
            input, weight, layout=layout, mat2_transposed=True
        )
        assert (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.int8
        )
        aten_layout = layout

        # options to tune from
        choices = (
            [aten__weight_int8pack_mm.bind((mat1, mat2, scale), aten_layout)]
            if use_aten_gemm_kernels()
            else []
        )

        # scale is applied as an epilogue, and the scale tensor is expanded (with a view op)
        # for broadcasting, as it's 1D.
        def _mul_epilogue(buf: torch.Tensor) -> Any:
            return create_epilogue_with_attr(
                buf, "mul", other=realize_inputs(expand(scale, layout.size))
            )

        if use_cpp_gemm_template(aten_layout, mat1, mat2, mat2_transposed=True):
            CppGemmTemplate.add_choices(
                choices,
                aten_layout,
                [mat1, mat2, scale],
                trans_w=True,
                epilogue_creator=_mul_epilogue,  # type: ignore[arg-type]
            )

        return autotune_select_algorithm(
            "_weight_int8pack_mm", choices, [mat1, mat2, scale], aten_layout
        )

    @register_lowering(aten._weight_int4pack_mm_for_cpu, type_promotion_kind=None)  # type: ignore[misc]
    def int4pack_mm_cpu(
        input: torch.Tensor,
        weight: torch.Tensor,
        qGroupSize: int,
        qScaleAndZeros: torch.Tensor,
        *,
        layout: Any = None,
    ) -> Any:
        _, _, _, layout, mat1, mat2 = mm_args(
            input, weight, layout=layout, use_4x2_dim=True, mat2_transposed=True
        )
        assert (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.uint8
        )
        group_size = V.graph.add_tensor_constant(
            torch.tensor(qGroupSize, dtype=torch.int64), name=None
        )
        aten_layout = layout

        # options to tune from
        choices = (
            [
                aten__weight_int4pack_mm_cpu.bind(
                    (mat1, mat2, group_size, qScaleAndZeros), aten_layout
                )
            ]
            if use_aten_gemm_kernels()
            else []
        )
        if (
            (config.max_autotune or config.max_autotune_gemm)
            and use_cpp_gemm_template(
                aten_layout,
                mat1,
                mat2,
                mat2_transposed=True,
                is_woq_int4=True,
                q_group_size=qGroupSize,
            )
            and mat2.get_layout().is_contiguous()
        ):
            # pyrefly: ignore [bad-specialization, missing-attribute, not-a-type]
            CppWoqInt4GemmTemplate[qGroupSize].add_choices(
                choices,
                aten_layout,
                [mat1, mat2, group_size, qScaleAndZeros],
            )

        # define functions to generate example inputs for weight and group size
        # otherwise, autotuner generates example inputs of all zeros for them
        def get_example_weight(x: torch._inductor.ir.IRNode) -> torch.Tensor:
            assert x.get_layout().is_contiguous()
            shape = x.get_size()
            device = x.get_device()
            return torch.randint(0, 255, shape, dtype=torch.uint8, device=device)

        input_gen_fns = {
            1: get_example_weight,  # packed weight
            2: lambda x: V.graph.constants[x.get_name()],  # group size
        }

        return autotune_select_algorithm(
            "_weight_int4pack_mm_for_cpu",
            choices,
            [mat1, mat2, group_size, qScaleAndZeros],
            aten_layout,
            input_gen_fns=input_gen_fns,
        )

    lowering.make_fallback(aten._dyn_quant_matmul_4bit)
    lowering.make_fallback(aten._dyn_quant_pack_4bit_weight)

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `register_quantized_ops`, `register_woq_mm_ops`, `int8pack_mm`, `_mul_epilogue`, `int4pack_mm_cpu`, `get_example_weight`

**Key imports**: logging, Any, torch, mm_args, config, lowering, CppGemmTemplate, CppWoqInt4GemmTemplate, create_epilogue_with_attr, expand, register_lowering, WeightInt4PackMatmul, use_aten_gemm_kernels, use_cpp_gemm_template


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `typing`: Any
- `torch`
- `torch._inductor.kernel.mm_common`: mm_args
- `.`: config, lowering
- `.codegen.cpp_gemm_template`: CppGemmTemplate, CppWoqInt4GemmTemplate
- `.codegen.cpp_utils`: create_epilogue_with_attr
- `.lowering`: expand, register_lowering
- `.mkldnn_ir`: WeightInt4PackMatmul
- `.utils`: use_aten_gemm_kernels, use_cpp_gemm_template
- `.virtualized`: V


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `quantized_lowerings.py_docs.md`
- **Keyword Index**: `quantized_lowerings.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `quantized_lowerings.py_docs.md_docs.md`
- **Keyword Index**: `quantized_lowerings.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
