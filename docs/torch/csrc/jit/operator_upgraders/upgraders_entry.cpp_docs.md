# Documentation: `torch/csrc/jit/operator_upgraders/upgraders_entry.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/operator_upgraders/upgraders_entry.cpp`
- **Size**: 7,108 bytes (6.94 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>

#include <ATen/core/stack.h>
#include <c10/macros/Export.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <string>
#include <unordered_map>

namespace torch::jit {

static std::unordered_map<std::string, std::string> kUpgradersEntryMap({
    {"logspace_0_8", R"SCRIPT(
def logspace_0_8(start: Union[int, float, complex], end: Union[int, float, complex], steps: Optional[int], base: float, *, dtype: Optional[int], layout: Optional[int],
                 device: Optional[Device], pin_memory: Optional[bool]):
  if (steps is None):
    return torch.logspace(start=start, end=end, steps=100, base=base, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
  return torch.logspace(start=start, end=end, steps=steps, base=base, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
)SCRIPT"},
    {"logspace_out_0_8", R"SCRIPT(
def logspace_out_0_8(start: Union[int, float, complex], end: Union[int, float, complex], steps: Optional[int], base: float, *, out: Tensor):
  if (steps is None):
    return torch.logspace(start=start, end=end, steps=100, base=base, out=out)
  return torch.logspace(start=start, end=end, steps=steps, base=base, out=out)
)SCRIPT"},
    {"linspace_0_7", R"SCRIPT(
def linspace_0_7(start: Union[int, float, complex], end: Union[int, float, complex], steps: Optional[int], *, dtype: Optional[int], layout: Optional[int],
                 device: Optional[Device], pin_memory: Optional[bool]):
  if (steps is None):
    return torch.linspace(start=start, end=end, steps=100, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
  return torch.linspace(start=start, end=end, steps=steps, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
)SCRIPT"},
    {"linspace_out_0_7", R"SCRIPT(
def linspace_out_0_7(start: Union[int, float, complex], end: Union[int, float, complex], steps: Optional[int], *, out: Tensor):
  if (steps is None):
    return torch.linspace(start=start, end=end, steps=100, out=out)
  return torch.linspace(start=start, end=end, steps=steps, out=out)
)SCRIPT"},
    {"div_Tensor_0_3", R"SCRIPT(
def div_Tensor_0_3(self: Tensor, other: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point()):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
)SCRIPT"},
    {"div_Tensor_mode_0_3", R"SCRIPT(
def div_Tensor_mode_0_3(self: Tensor, other: Tensor, *, rounding_mode: Optional[str]=None) -> Tensor:
  return self.divide(other, rounding_mode=rounding_mode)
)SCRIPT"},
    {"div_Scalar_0_3", R"SCRIPT(
def div_Scalar_0_3(self: Tensor, other: number) -> Tensor:
  if (self.is_floating_point() or isinstance(other, float)):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
)SCRIPT"},
    {"div_Scalar_mode_0_3", R"SCRIPT(
def div_Scalar_mode_0_3(self: Tensor, other: number, *, rounding_mode: Optional[str]=None) -> Tensor:
  return self.divide(other, rounding_mode=rounding_mode)
)SCRIPT"},
    {"div_out_0_3", R"SCRIPT(
def div_out_0_3(self: Tensor, other: Tensor, *, out: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point() or out.is_floating_point()):
    return self.true_divide(other, out=out)
  return self.divide(other, rounding_mode='trunc', out=out)
)SCRIPT"},
    {"div_out_mode_0_3", R"SCRIPT(
def div_out_mode_0_3(self: Tensor, other: Tensor, *, rounding_mode: Optional[str]=None, out: Tensor) -> Tensor:
  return self.divide(other, rounding_mode=rounding_mode, out=out)
)SCRIPT"},
    {"div__Tensor_0_3", R"SCRIPT(
def div__Tensor_0_3(self: Tensor, other: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point()):
    return self.true_divide_(other)
  return self.divide_(other, rounding_mode='trunc')
)SCRIPT"},
    {"div__Tensor_mode_0_3", R"SCRIPT(
def div__Tensor_mode_0_3(self: Tensor, other: Tensor, *, rounding_mode: Optional[str]=None) -> Tensor:
  return self.divide_(other, rounding_mode=rounding_mode)
)SCRIPT"},
    {"div__Scalar_0_3", R"SCRIPT(
def div__Scalar_0_3(self: Tensor, other: number) -> Tensor:
  if (self.is_floating_point() or isinstance(other, float)):
    return self.true_divide_(other)
  return self.divide_(other, rounding_mode='trunc')
)SCRIPT"},
    {"div__Scalar_mode_0_3", R"SCRIPT(
def div__Scalar_mode_0_3(self: Tensor, other: number, *, rounding_mode: Optional[str]=None) -> Tensor:
  return self.divide_(other, rounding_mode=rounding_mode)
)SCRIPT"},
    {"full_names_0_4", R"SCRIPT(
def full_names_0_4(size:List[int], fill_value:number, *, names:Optional[List[str]]=None,
                   dtype:Optional[int]=None, layout:Optional[int]=None, device:Optional[Device]=None,
                   pin_memory:Optional[bool]=None) -> Tensor:
  return torch.full(size, fill_value, names=names, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
)SCRIPT"},
    {"full_0_4", R"SCRIPT(
def full_0_4(size:List[int], fill_value:number, *, dtype:Optional[int]=None,
             layout:Optional[int]=None, device:Optional[Device]=None,
             pin_memory:Optional[bool]=None) -> Tensor:
  if dtype is None:
    fill_value = float(fill_value)
  return torch.full(size, fill_value, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
)SCRIPT"},
    {"full_out_0_4", R"SCRIPT(
def full_out_0_4(size:List[int], fill_value:number, *, out:Tensor) -> Tensor:
  return torch.full(size, fill_value, out=out)
)SCRIPT"},
    {"gelu_0_9", R"SCRIPT(
def gelu_0_9(self: Tensor) -> Tensor:
  return torch.gelu(self, approximate='none')
)SCRIPT"},
    {"gelu_out_0_9", R"SCRIPT(
def gelu_out_0_9(self: Tensor, *, out: Tensor) -> Tensor:
  return torch.gelu(self, approximate='none', out=out)
)SCRIPT"},
});

std::shared_ptr<Graph> create_upgrader_graph(
    const std::string& upgrader_name,
    const std::string& upgrader_body) {
  auto cu = std::make_shared<CompilationUnit>();
  cu->define(std::nullopt, upgrader_body, nativeResolver(), nullptr);
  Function& jitFunc = cu->get_function(upgrader_name);
  GraphFunction& graphFunction = toGraphFunction(jitFunc);
  return graphFunction.graph();
}

std::unordered_map<std::string, std::shared_ptr<Graph>>
generate_upgraders_graph() {
  std::unordered_map<std::string, std::shared_ptr<Graph>> populate_content;
  for (const auto& entry : kUpgradersEntryMap) {
    auto upgrader_graph = create_upgrader_graph(entry.first, entry.second);
    populate_content.insert(std::make_pair(entry.first, upgrader_graph));
  }
  return populate_content;
}

void populate_upgraders_graph_map() {
  if (!is_upgraders_map_populated()) {
    auto graphs = generate_upgraders_graph();
    populate_upgraders_map(std::move(graphs));
  }
}

std::unordered_map<std::string, std::string> get_upgraders_entry_map() {
  return kUpgradersEntryMap;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 22 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/operator_upgraders`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/operator_upgraders/upgraders_entry.h`
- `ATen/core/stack.h`
- `c10/macros/Export.h`
- `torch/csrc/jit/api/compilation_unit.h`
- `torch/csrc/jit/api/function_impl.h`
- `torch/csrc/jit/frontend/ir_emitter.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/operator_upgraders/upgraders.h`
- `torch/csrc/jit/serialization/export_bytecode.h`
- `string`
- `unordered_map`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/jit/operator_upgraders`):

- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`version_map.cpp_docs.md`](./version_map.cpp_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`version_map.h_docs.md`](./version_map.h_docs.md)
- [`upgraders.cpp_docs.md`](./upgraders.cpp_docs.md)
- [`upgraders.h_docs.md`](./upgraders.h_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`upgraders_entry.h_docs.md`](./upgraders_entry.h_docs.md)


## Cross-References

- **File Documentation**: `upgraders_entry.cpp_docs.md`
- **Keyword Index**: `upgraders_entry.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
