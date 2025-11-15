# Documentation: `test/dynamo/test_fx_annotate.py`

## File Metadata

- **Path**: `test/dynamo/test_fx_annotate.py`
- **Size**: 13,944 bytes (13.62 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch.fx.traceback as fx_traceback
import torch.utils.checkpoint
from torch._dynamo.test_case import run_tests
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


class AnnotateTests(torch._dynamo.test_case.TestCase):
    def test_annotations(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                with fx_traceback.annotate({"pp_stage": 0}):
                    with fx_traceback.annotate({"fdsp_bucket": 0}):
                        sin = torch.sin(x)
                    sub = sin - 2
                    with fx_traceback.annotate({"cuda_stream": 2, "fsdp_bucket": 1}):
                        mul = sub * 2
                div = mul / 3
                return div

        m = Mod()
        backend = AotEagerAndRecordGraphs()
        opt_m = torch.compile(m, backend=backend, fullgraph=True)
        x = torch.randn(10, requires_grad=True)
        opt_m(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """\
('placeholder', 'l_x_', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sin', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sub', {'pp_stage': 0})
('call_function', 'mul', {'pp_stage': 0, 'cuda_stream': 2, 'fsdp_bucket': 1})""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """\
('call_function', 'sin', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sub', {'pp_stage': 0})
('call_function', 'mul', {'pp_stage': 0, 'cuda_stream': 2, 'fsdp_bucket': 1})""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('call_function', 'mul_1', {'pp_stage': 0, 'cuda_stream': 2, 'fsdp_bucket': 1})
('call_function', 'cos', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'mul_2', {'pp_stage': 0, 'fdsp_bucket': 0})""",  # noqa: B950
        )

    def test_activation_checkpointing(self):
        @checkpoint_wrapper
        def gn(x):
            return torch.sin(x)

        def fn(x):
            with fx_traceback.annotate({"ac_sin": 0}):
                ac = gn(x)
            return torch.sigmoid(ac)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(10, requires_grad=True)
        opt_fn(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """\
('placeholder', 'l_x_', {'ac_sin': 0})
('get_attr', 'wrap_body_0', {'ac_sin': 0})
[('placeholder', 'l_x_', {'ac_sin': 0}), ('call_function', 'sin', {'ac_sin': 0}), ('output', 'output', {'ac_sin': 0})]
('call_function', 'tag_activation_checkpoint', {'ac_sin': 0})
('call_function', 'ac', {'ac_sin': 0})""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """('call_function', 'sin', {'ac_sin': 0})""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('call_function', 'cos', {'ac_sin': 0})
('call_function', 'mul', {'ac_sin': 0})""",  # noqa: B950
        )

    def test_activation_checkpointing_annotation_inside(self):
        @checkpoint_wrapper
        def gn(x):
            x = x + 1
            with fx_traceback.annotate({"stage": 0}):
                p = torch.sin(x)
            return p + 1

        def fn(x):
            ac = gn(x)
            return torch.sigmoid(ac)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(10, requires_grad=True)
        opt_fn(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """[('call_function', 'p', {'stage': 0})]""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """('call_function', 'sin', {'stage': 0})""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('call_function', 'cos', {'stage': 0})
('call_function', 'mul', {'stage': 0})""",  # noqa: B950
        )

    @requires_cuda_and_triton
    def test_ac_flex_attention(self):
        def _squared(score, b, h, m, n):
            return score * score

        def mask_mod(b, h, q, k):
            return q >= 0

        a = 12
        b = 64
        block_mask = create_block_mask(mask_mod, None, None, a * b, a * b)

        def gn(x: torch.Tensor):
            with fx_traceback.annotate({"compile_inductor": 0}):
                return flex_attention(
                    x, x, x, block_mask=block_mask, score_mod=_squared
                )

        def fn(x):
            x = torch.sin(x)
            x = gn(x)
            return torch.cos(x)

        x = torch.randn(
            1,
            1,
            a * b,
            b,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        opt_fn(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """\
('placeholder', 'l_gn_closure_1_cell_contents_kv_indices', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_kv_num_blocks', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_full_kv_num_blocks', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_full_kv_indices', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_q_num_blocks', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_q_indices', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_full_q_num_blocks', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_full_q_indices', {'compile_inductor': 0})
('get_attr', 'score_mod_0', {'compile_inductor': 0})
[('placeholder', 'child', {'compile_inductor': 0}), ('placeholder', 'child_1', {'compile_inductor': 0}), ('placeholder', 'child_2', {'compile_inductor': 0}), ('placeholder', 'child_3', {'compile_inductor': 0}), ('placeholder', 'child_4', {'compile_inductor': 0}), ('call_function', 'mul', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('get_attr', 'mask_fn_0', {'compile_inductor': 0})
[('placeholder', 'child', {'compile_inductor': 0}), ('placeholder', 'child_1', {'compile_inductor': 0}), ('placeholder', 'child_2', {'compile_inductor': 0}), ('placeholder', 'child_3', {'compile_inductor': 0}), ('call_function', 'ge', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('call_function', 'flex_attention', {'compile_inductor': 0})
('call_function', 'out', {'compile_inductor': 0})""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """\
('get_attr', 'sdpa_score0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('placeholder', 'arg4_1', {'compile_inductor': 0}), ('call_function', 'mul', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('get_attr', 'sdpa_mask0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('call_function', 'ge', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('call_function', 'flex_attention', {'compile_inductor': 0})
('call_function', 'getitem', {'compile_inductor': 0})
('call_function', 'getitem_1', {'compile_inductor': 0})
('call_function', 'detach_1', {'compile_inductor': 0})
('call_function', 'detach_3', {'compile_inductor': 0})""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('placeholder', 'getitem', {'compile_inductor': 0})
('placeholder', 'detach_3', {'compile_inductor': 0})
('call_function', 'zeros', {'compile_inductor': 0})
('call_function', 'detach', {'compile_inductor': 0})
('call_function', 'detach_2', {'compile_inductor': 0})
('get_attr', 'fw_graph0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('placeholder', 'arg4_1', {'compile_inductor': 0}), ('call_function', 'mul', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('get_attr', 'joint_graph0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('placeholder', 'arg4_1', {'compile_inductor': 0}), ('placeholder', 'arg5_1', {'compile_inductor': 0}), ('call_function', 'mul_1', {'compile_inductor': 0}), ('call_function', 'mul_2', {'compile_inductor': 0}), ('call_function', 'add', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('get_attr', 'mask_graph0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('call_function', 'ge', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('call_function', 'flex_attention_backward', {'compile_inductor': 0})
('call_function', 'getitem_3', {'compile_inductor': 0})
('call_function', 'getitem_4', {'compile_inductor': 0})
('call_function', 'getitem_5', {'compile_inductor': 0})""",  # noqa: B950
        )

    def test_as_decorator(self):
        class Mod(torch.nn.Module):
            @fx_traceback.annotate({"fdsp_bucket": 0})
            def sin(self, x):
                return torch.sin(x)

            def forward(self, x):
                with fx_traceback.annotate({"pp_stage": 0}):
                    sin = self.sin(x)
                    sub = sin - 2
                    mul = sub * 2
                div = mul / 3
                return div

        m = Mod()
        backend = AotEagerAndRecordGraphs()
        opt_m = torch.compile(m, backend=backend, fullgraph=True)
        x = torch.randn(10, requires_grad=True)
        m(x)
        opt_m(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """\
('placeholder', 'l_x_', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sin', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sub', {'pp_stage': 0})
('call_function', 'mul', {'pp_stage': 0})""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """\
('call_function', 'sin', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sub', {'pp_stage': 0})
('call_function', 'mul', {'pp_stage': 0})""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('call_function', 'mul_1', {'pp_stage': 0})
('call_function', 'cos', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'mul_2', {'pp_stage': 0, 'fdsp_bucket': 0})""",  # noqa: B950
        )

    def test_graph_break(self):
        def fn(x):
            with torch.fx.traceback.annotate({"pp_stage": 0}):
                x = torch.sin(x)
                torch._dynamo.graph_break()
                x = torch.cos(x)
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(10, requires_grad=True)
        self.assertEqual(fn(x), opt_fn(x))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""\('placeholder', 'l_x_', {'pp_stage': 0, 'fdsp_bucket': 0})('call_function', 'sin', {'pp_stage': 0, 'fdsp_bucket': 0})('call_function', 'sub', {'pp_stage': 0})

This Python file contains 3 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AnnotateTests`, `Mod`, `Mod`

**Functions defined**: `checkpoint_wrapper`, `inner`, `test_annotations`, `forward`, `test_activation_checkpointing`, `gn`, `fn`, `test_activation_checkpointing_annotation_inside`, `gn`, `fn`, `test_ac_flex_attention`, `_squared`, `mask_mod`, `gn`, `fn`, `test_as_decorator`, `sin`, `forward`, `test_graph_break`, `fn`

**Key imports**: torch, torch._dynamo.test_case, torch.fx.traceback as fx_traceback, torch.utils.checkpoint, run_tests, AotEagerAndRecordGraphs, create_block_mask, flex_attention, requires_cuda_and_triton


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dynamo.test_case`
- `torch.fx.traceback as fx_traceback`
- `torch.utils.checkpoint`
- `torch._dynamo.testing`: AotEagerAndRecordGraphs
- `torch.nn.attention.flex_attention`: create_block_mask, flex_attention
- `torch.testing._internal.triton_utils`: requires_cuda_and_triton


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

This is a test file. Run it with:

```bash
python test/dynamo/test_fx_annotate.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_fx_annotate.py_docs.md`
- **Keyword Index**: `test_fx_annotate.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
