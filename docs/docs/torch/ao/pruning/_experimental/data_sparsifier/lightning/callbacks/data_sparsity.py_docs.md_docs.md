# Documentation: `docs/torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks/data_sparsity.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks/data_sparsity.py_docs.md`
- **Size**: 10,180 bytes (9.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks/data_sparsity.py`

## File Metadata

- **Path**: `torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks/data_sparsity.py`
- **Size**: 6,595 bytes (6.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections import defaultdict
from copy import deepcopy
from typing import Any, TYPE_CHECKING

import pytorch_lightning as pl  # type: ignore[import]

from ._data_sparstity_utils import (
    _attach_model_to_data_sparsifier,
    _get_valid_name,
    _log_sparsified_level,
)


if TYPE_CHECKING:
    import torch


class PostTrainingDataSparsity(pl.callbacks.Callback):
    """Lightning callback that enables post-training sparsity.

    This callback aims to sparsify the model inside lightning module after training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_class (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this class is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            once the training completes.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

    Hooks implemented:
        on_fit_end()
            1. copies the model and attaches it to the sparsifier
            2. sparsier step() is called
            3. squashes the mask()
    """

    def __init__(self, data_sparsifier_class, data_sparsifier_args):
        super().__init__()
        self.data_sparsifier_class = data_sparsifier_class
        self.data_sparsifier_args = data_sparsifier_args
        self.data_sparsifier: Any = None
        self.sparsified: torch.nn.Module | None = None

    def on_fit_end(self, trainer, pl_module) -> None:
        self.sparsified = deepcopy(pl_module.model).eval()
        self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)

        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)

        self.data_sparsifier.step()

        self.data_sparsifier.squash_mask()  # currently squashes params for all mask

        _log_sparsified_level(self.sparsified, self.data_sparsifier)


class TrainingAwareDataSparsity(pl.callbacks.Callback):
    """Lightning callback that enables in-training sparsity.

    This callback aims to sparsify the model inside lightning module during training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_class (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this class is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            when the training starts.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

        data_scheduler_class (some implemented class of BaseDataScheduler)
            The data scheduler of this class is created when the training starts
            Note: Objects should not be passed in here as they are created
            when the training starts.

        data_scheduler_args(Dict)
            Dictionary of args to be passed to the data scheduler.
            **Note: data_sparsifier arg should be ignored as the recipe
            creates and pass sparsifier object into the class**

    Hooks implemented:
        on_train_start()
            Data sparsifier and scheduler objects are created.
            Pytorch model attached to the sparsifier

        on_train_epoch_start()
            Loads the state_dict of the data sparsifier

        on_train_epoch_end()
            1. Copies the model and attaches it to the sparsifier
            2. sparsifier step() and scheduler step()
            3. Dump state_dict of the current sparsifier

        on_train_end()
            squash mask
    """

    def __init__(
        self,
        data_sparsifier_class,
        data_sparsifier_args,
        data_scheduler_class,
        data_scheduler_args,
    ):
        super().__init__()
        # data sparsifier objects
        self.data_sparsifier_class = data_sparsifier_class
        self.data_sparsifier_args = data_sparsifier_args

        # scheduler objects
        self.data_scheduler_class = data_scheduler_class
        self.data_scheduler_args = data_scheduler_args

        # fields
        self.data_sparsifier: Any = None
        self.data_scheduler: Any = None
        self.sparsified: torch.nn.Module | None = None

        self.data_sparsifier_state_dict: Any = None

    def on_train_start(self, trainer, pl_module) -> None:
        # create sparsifier
        self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)
        self.sparsified = deepcopy(pl_module.model)

        _attach_model_to_data_sparsifier(
            self.sparsified, self.data_sparsifier
        )  # just to populate the base_sl in the scheduler

        # create scheduler
        args = deepcopy(self.data_scheduler_args)
        args["data_sparsifier"] = self.data_sparsifier
        self.data_scheduler = self.data_scheduler_class(**args)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.data_sparsifier_state_dict is None:
            return  # probably first epoch

        # load the existing config for each data
        self.data_sparsifier.load_state_dict(self.data_sparsifier_state_dict)

    def __create_config_based_on_state(self, pl_module):
        config: dict = defaultdict()
        if self.data_sparsifier_state_dict is None:
            return config
        for name, _ in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            config[valid_name] = self.data_sparsifier.data_groups[valid_name]

        return config

    def on_train_epoch_end(self, trainer, pl_module):
        self.sparsified = deepcopy(pl_module.model)
        config = self.__create_config_based_on_state(pl_module)

        # attach model to the data sparsifier
        _attach_model_to_data_sparsifier(
            self.sparsified, self.data_sparsifier, config=config
        )
        self.data_sparsifier.step()
        self.data_scheduler.step()

        self.data_sparsifier_state_dict = self.data_sparsifier.state_dict()

    def on_train_end(self, trainer, pl_module):
        self.data_sparsifier.squash_mask()

```



## High-Level Overview

"""Lightning callback that enables post-training sparsity.    This callback aims to sparsify the model inside lightning module after training.    **Note that the model is copied and then sparsified, so the existing model is not modified**    The sparsified model can be used for comparison and can be accessed using        <callback_obj>.sparsified    Args:        data_sparsifier_class (some implemented class of BaseDataSparsifier)            The data sparsifier object of this class is created when the            training starts.            Note: Objects should not be passed in here as they are created            once the training completes.        data_sparsifier_args (Dict)            Dictionary of args to be passed to the data sparsifier.            Note: data_list arg should be ignored    Hooks implemented:        on_fit_end()            1. copies the model and attaches it to the sparsifier            2. sparsier step() is called            3. squashes the mask()

This Python file contains 11 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PostTrainingDataSparsity`, `TrainingAwareDataSparsity`

**Functions defined**: `__init__`, `on_fit_end`, `__init__`, `on_train_start`, `on_train_epoch_start`, `__create_config_based_on_state`, `on_train_epoch_end`, `on_train_end`

**Key imports**: defaultdict, deepcopy, Any, TYPE_CHECKING, pytorch_lightning as pl  , torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: defaultdict
- `copy`: deepcopy
- `typing`: Any, TYPE_CHECKING
- `pytorch_lightning as pl  `
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`_data_sparstity_utils.py_docs.md`](./_data_sparstity_utils.py_docs.md)


## Cross-References

- **File Documentation**: `data_sparsity.py_docs.md`
- **Keyword Index**: `data_sparsity.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks`):

- [`_data_sparstity_utils.py_kw.md_docs.md`](./_data_sparstity_utils.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`data_sparsity.py_kw.md_docs.md`](./data_sparsity.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`README.md_kw.md_docs.md`](./README.md_kw.md_docs.md)
- [`_data_sparstity_utils.py_docs.md_docs.md`](./_data_sparstity_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `data_sparsity.py_docs.md_docs.md`
- **Keyword Index**: `data_sparsity.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
