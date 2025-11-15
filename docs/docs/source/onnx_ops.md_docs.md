# Documentation: `docs/source/onnx_ops.md`

## File Metadata

- **Path**: `docs/source/onnx_ops.md`
- **Size**: 3,556 bytes (3.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# torch.onnx.ops

```{eval-rst}
.. automodule:: torch.onnx.ops
```

## Symbolic Operators

Operators that can be used to create any ONNX ops in the FX graph symbolically.
These operators do not do actual computation. It's recommended that you used them
inside an ``if torch.onnx.is_in_onnx_export`` block.

```{eval-rst}
.. autofunction:: torch.onnx.ops.symbolic
.. autofunction:: torch.onnx.ops.symbolic_multi_out
```

## ONNX Operators

The following operators are implemented as native PyTorch ops and can be exported as
ONNX operators. They can be used natively in an ``nn.Module``.

For example, you can define a module:

```py
class Model(torch.nn.Module):
    def forward(
        self, input_data, cos_cache_data, sin_cache_data, position_ids_data
    ):
        return torch.onnx.ops.rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            position_ids_data,
        )
```

and export it to ONNX using:

```py
input_data = torch.rand(2, 3, 4, 8)
position_ids_data = torch.randint(0, 50, (2, 3)).long()
sin_cache_data = torch.rand(50, 4)
cos_cache_data = torch.rand(50, 4)
dynamic_shapes = {
    "input_data": {0: torch.export.Dim.DYNAMIC},
    "cos_cache_data": None,
    "sin_cache_data": None,
    "position_ids_data": {0: torch.export.Dim.DYNAMIC},
}
onnx_program = torch.onnx.export(
    model,
    (input_data, cos_cache_data, sin_cache_data, position_ids_data),
    dynamic_shapes=dynamic_shapes,
    dynamo=True,
    opset_version=23,
)
```

Printing the ONNX program will show the ONNX operators used in the graph:

```
<...>

graph(
    name=main_graph,
    inputs=(
        %"input_data"<FLOAT,[s0,3,4,8]>,
        %"cos_cache_data"<FLOAT,[50,4]>,
        %"sin_cache_data"<FLOAT,[50,4]>,
        %"position_ids_data"<INT64,[s0,3]>
    ),
    outputs=(
        %"rotary_embedding"<FLOAT,[s0,3,4,8]>
    ),
) {
    0 |  # rotary_embedding
        %"rotary_embedding"<FLOAT,[s0,3,4,8]> ⬅️ ::RotaryEmbedding(%"input_data", %"cos_cache_data", %"sin_cache_data", %"position_ids_data")
    return %"rotary_embedding"<FLOAT,[s0,3,4,8]>
}
```

with the corresponding ``ExportedProgram``:

ExportedProgram:

```py
class GraphModule(torch.nn.Module):
    def forward(self, input_data: "f32[s0, 3, 4, 8]", cos_cache_data: "f32[50, 4]", sin_cache_data: "f32[50, 4]", position_ids_data: "i64[s0, 3]"):
        rotary_embedding: "f32[s0, 3, 4, 8]" = torch.ops.onnx.RotaryEmbedding.opset23(input_data, cos_cache_data, sin_cache_data, position_ids_data);  input_data = cos_cache_data = sin_cache_data = position_ids_data = None
        return (rotary_embedding,)
```

```{eval-rst}
.. autofunction:: torch.onnx.ops.rotary_embedding
.. autofunction:: torch.onnx.ops.attention
```

## ONNX to ATen Decomposition Table

You can use {func}`torch.onnx.ops.aten_decompositions` to obtain a decomposition table
to decompose ONNX operators defined above to ATen operators.

```py
class Model(torch.nn.Module):
    def forward(
        self, input_data, cos_cache_data, sin_cache_data, position_ids_data
    ):
        return torch.onnx.ops.rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            position_ids_data,
        )

model = Model()

ep = torch.export.export(
    model,
    (input_data, cos_cache_data, sin_cache_data, position_ids_data),
)
# The program can be decomposed into aten ops
ep_decomposed = ep.run_decompositions(torch.onnx.ops.aten_decompositions())
```

```{eval-rst}
.. autofunction:: torch.onnx.ops.aten_decompositions
```

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`docs/source`):

- [`torch.compiler_troubleshooting.md_docs.md`](./torch.compiler_troubleshooting.md_docs.md)
- [`torch.compiler_aot_inductor_debugging_guide.md_docs.md`](./torch.compiler_aot_inductor_debugging_guide.md_docs.md)
- [`mtia.memory.md_docs.md`](./mtia.memory.md_docs.md)
- [`torch.compiler_get_started.md_docs.md`](./torch.compiler_get_started.md_docs.md)
- [`torch.compiler_dynamo_deepdive.md_docs.md`](./torch.compiler_dynamo_deepdive.md_docs.md)
- [`mtia.mtia_graph.md_docs.md`](./mtia.mtia_graph.md_docs.md)
- [`hub.md_docs.md`](./hub.md_docs.md)
- [`torch_nccl_environment_variables.md_docs.md`](./torch_nccl_environment_variables.md_docs.md)
- [`optim.md_docs.md`](./optim.md_docs.md)
- [`torch.compiler_aot_inductor.md_docs.md`](./torch.compiler_aot_inductor.md_docs.md)


## Cross-References

- **File Documentation**: `onnx_ops.md_docs.md`
- **Keyword Index**: `onnx_ops.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
