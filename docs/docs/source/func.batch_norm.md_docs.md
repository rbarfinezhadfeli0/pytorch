# Documentation: `docs/source/func.batch_norm.md`

## File Metadata

- **Path**: `docs/source/func.batch_norm.md`
- **Size**: 2,819 bytes (2.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Patching Batch Norm

## What's happening?
Batch Norm requires in-place updates to running_mean and running_var of the same size as the input.
Functorch does not support inplace update to a regular tensor that takes in a batched tensor (i.e.
`regular.add_(batched)` is not allowed). So when vmapping over a batch of inputs to a single module,
we end up with this error

## How to fix
One of the best supported ways is to switch BatchNorm for GroupNorm. Options 1 and 2 support this

All of these options assume that you don't need running stats. If you're using a module this means
that it's assumed you won't use batch norm in evaluation mode. If you have a use case that involves
running batch norm with vmap in evaluation mode, please file an issue

### Option 1: Change the BatchNorm
If you want to change for GroupNorm, anywhere that you have BatchNorm, replace it with:

```python
BatchNorm2d(C, G, track_running_stats=False)
```

Here `C` is the same `C` as in the original BatchNorm. `G` is the number of groups to
break `C` into. As such, `C % G == 0` and as a fallback, you can set `C == G`, meaning
each channel will be treated separately.

If you must use BatchNorm and you've built the module yourself, you can change the module to
not use running stats. In other words, anywhere that there's a BatchNorm module, set the
`track_running_stats` flag to be False

```python
BatchNorm2d(64, track_running_stats=False)
```

### Option 2: torchvision parameter
Some torchvision models, like resnet and regnet, can take in a `norm_layer` parameter. These are
often defaulted to be BatchNorm2d if they've been defaulted.

Instead you can set it to be GroupNorm.

```python
import torchvision
from functools import partial
torchvision.models.resnet18(norm_layer=lambda c: GroupNorm(num_groups=g, c))
```

Here, once again, `c % g == 0` so as a fallback, set `g = c`.

If you are attached to BatchNorm, be sure to use a version that doesn't use running stats

```python
import torchvision
from functools import partial
torchvision.models.resnet18(norm_layer=partial(BatchNorm2d, track_running_stats=False))
```

### Option 3: functorch's patching
functorch has added some functionality to allow for quick, in-place patching of the module to not
use running stats. Changing the norm layer is more fragile, so we have not offered that. If you
have a net where you want the BatchNorm to not use running stats, you can run
`replace_all_batch_norm_modules_` to update the module in-place to not use running stats

```python
from torch.func import replace_all_batch_norm_modules_
replace_all_batch_norm_modules_(net)
```

### Option 4: eval mode
When run under eval mode, the running_mean and running_var will not be updated. Therefore, vmap can support this mode

```python
model.eval()
vmap(model)(x)
model.train()
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

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

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

- **File Documentation**: `func.batch_norm.md_docs.md`
- **Keyword Index**: `func.batch_norm.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
