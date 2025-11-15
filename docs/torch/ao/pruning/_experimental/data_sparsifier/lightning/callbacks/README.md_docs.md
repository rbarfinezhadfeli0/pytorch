# Documentation: `torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks/README.md`

## File Metadata

- **Path**: `torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks/README.md`
- **Size**: 3,261 bytes (3.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This is a markdown documentation that is part of the PyTorch project.

## Original Source

```markdown
# Lightning callbacks for data sparsifier and scheduler

**These are callback scripts for lightning and does not introduce pytorch lightning dependency on PyTorch.**

## Introduction
Callbacks for PytorchLightning that specifies on when and how to sparsify the data weights of the model.

## Types of Data Sparsity Callbacks
There are 2 types of data sparsity callbacks
1. **Post Training data sparsifier callback**: Sparsification of the model parameters *post* training.

2. **Training Aware data sparsifier callback**: Sparsification of the model parameters *during* training.

## API Design
1. `PostTrainingDataSparsity`: callback class that sparsifies the model parameters post training. Accepts
    1.  `data_sparsifier_class`: class/type of data sparsifier that needs to be used. Only the class should be passed, the data sparsifier object
    will be created internally and will be attached to the model by the callback whenever necessary.
    2. `data_sparsifier_args`: the arguments/config for the data sparsifier constructor that will be used while creating the object.

    Example:
    ```
    from data_sparsity import PostTrainingDataSparsity
    sparsifier_args = {
        'sparsity_level': 0.5,
        'sparse_block_shape': (1, 4),
        'zeros_per_block': 4
    }
    pt_callback = PostTrainingDataSparsity(data_sparsifier_class=DataNormSparsifier, data_sparsifier_args=sparsifier_args)
    ```

2. `TrainingAwareDataSparsity`: callback class to sparsify model during training. In addition to `data_sparsifier_class` and `data_sparsifier_args`,
    also accepts
    1. `data_scheduler_class`: class/type of data scheduler to schedule the sparsity levels during training. Only the class should be passed, the object
    will be created internally whenever necessary.
    2. `data_scheduler_args`: the arguments/config for the data scheduler constructor that will be used while creating the object.

    Example:

    ```
    from data_sparsity import TrainingAwareDataSparsity
    sparsifier_args = {
        'sparsity_level': 0.5,
        'sparse_block_shape': (1, 4),
        'zeros_per_block': 4
    }
    scheduler_args = {
        'gamma': 2,
        'step_size': 1
    }

    ta_callback = TrainingAwareDataSparsity(
        data_sparsifier_class=DataNormSparsifier,
        data_sparsifier_args=sparsifier_args,
        data_scheduler_class=StepSLScheduler,
        data_scheduler_args=scheduler_args
    )
    ```

**Note:**
1. The model is copied and then sparsified, so the existing model is not modified.
2. The sparsified model can be accessed using `sparsified` attribute and can be used for comparison with the original version.
3. The data sparsifier/scheduler object will be created internally and will be attached to the model by the callback whenever necessary.

## Usage
```
pl_module = SomePLModule()  # pl_module.model should specify the pytorch model

ds_callback = SomeDataSparsifierCallback(data_sparsifier_class=..., data_sparsifier_args=..., ...)  # add scheduler if TrainingAwareDataSparsifier
trainer = Trainer(callbacks=[ds_callback])

trainer.fit(pl_module, train_data_loader, val_data_loader)

# NOTE: pl_module.model is not sparsified

# access sparsified model
sparsified_model = ds_callback.sparsified
```

```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`data_sparsity.py_docs.md`](./data_sparsity.py_docs.md)
- [`_data_sparstity_utils.py_docs.md`](./_data_sparstity_utils.py_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md`
- **Keyword Index**: `README.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
