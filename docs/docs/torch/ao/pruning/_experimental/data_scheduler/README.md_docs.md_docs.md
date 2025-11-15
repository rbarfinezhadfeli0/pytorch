# Documentation: `docs/torch/ao/pruning/_experimental/data_scheduler/README.md_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/pruning/_experimental/data_scheduler/README.md_docs.md`
- **Size**: 4,996 bytes (4.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/pruning/_experimental/data_scheduler/README.md`

## File Metadata

- **Path**: `torch/ao/pruning/_experimental/data_scheduler/README.md`
- **Size**: 3,012 bytes (2.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This is a markdown documentation that is part of the PyTorch project.

## Original Source

```markdown
# Data Scheduler
## Intro
The data scheduler is used to control the update of the data sparsification parameters and works specifically with the data sparsifier class.
This class controls a specific config param (specified by the `schedule_param` argument) of
the data sparsifier class and varies it across the training process (or across time).

## API details
`BaseDataScheduler`: base class with abstract method `get_schedule_param` that computes the data sparsification parameter for all the data. The constructor accepts
1. `data_sparsifier`: The data sparsifier object whose parameter will be scheduled.
2. `schedule_param` : a specific config of the passed data sparsifier that needs to be scheduled/varied.

`get_last_param`: gets the last scheduled parameter. Basically, a dictionary of name (of data) to schedule_param value mapping.

`step`: Applies the `get_schedule_param` logic every epoch/step depending on when it is called. This should always be called after the `sparsifier.step()` has been called.

## Write your own data scheduler
The custom data scheduler must be inherit from the `BaseDataScheduler` class and should have the `get_schedule_param()` function implemented. For example, that gradually multiplies the sparsity level by `gamma` every epoch.
It also takes an argument `threshold_sl` which when reached does not increase further.

```
class GammaScheduler(BaseDataScheduler):
    def __init__(self, data_sparsifier, gamma, threshold_sl):
        super().__init__(data_sparsifier, "sparsity_level")
        self.gamma = gamma
        self.threshold_sl = threshold_sl

    def get_schedule_param(self):
        if self.last_epoch > 0:
            return {name: min(self.threshold_sl, config["sparsity_level"] * self.gamma) for name, config in self.data_sparsifier.data_groups.items()}
        else:
            return {name: 0.0 for name, config in self.data_sparsifier.data_groups.items()}
```

## Using data scheduler with data sparsifier
Suppose the need is to vary data sparsity levels (or any sparsity `param`) during training, then a custom data scheduler can be implemented and used along with the data sparsifier.

Example:

```
model = SomeModel()
optimizer = SomeOptimizer(model.parameters(), lr=...)
data_sparsifier = SomeDataSparsifier(...)


data_scheduler = SomeDataScheduler(data_sparsifier, ...)


data_name = 'train_data'

for epoch in range(EPOCHS):
    for input, target in dataset:
        input = data_sparsifier.add_data(name=data_name, data=input)

        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        data_sparsifier.step()

    data_scheduler.step()
```

### Note:
1. `get_schedule_param()` should return a dictionary wherein the keys are the names of the data and the values are the corresponding values of the `schedule_param` for the next step.
2. It is the responsibility of the `BaseDataScheduler` to call the `get_schedule_param()` when necessary.

```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/ao/pruning/_experimental/data_scheduler`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/pruning/_experimental/data_scheduler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/ao/pruning/_experimental/data_scheduler`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`base_data_scheduler.py_docs.md`](./base_data_scheduler.py_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md`
- **Keyword Index**: `README.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/pruning/_experimental/data_scheduler`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/pruning/_experimental/data_scheduler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/ao/pruning/_experimental/data_scheduler`):

- [`base_data_scheduler.py_kw.md_docs.md`](./base_data_scheduler.py_kw.md_docs.md)
- [`base_data_scheduler.py_docs.md_docs.md`](./base_data_scheduler.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`README.md_kw.md_docs.md`](./README.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md_docs.md`
- **Keyword Index**: `README.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
