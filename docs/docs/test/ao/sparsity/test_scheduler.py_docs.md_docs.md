# Documentation: `docs/test/ao/sparsity/test_scheduler.py_docs.md`

## File Metadata

- **Path**: `docs/test/ao/sparsity/test_scheduler.py_docs.md`
- **Size**: 10,340 bytes (10.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/ao/sparsity/test_scheduler.py`

## File Metadata

- **Path**: `test/ao/sparsity/test_scheduler.py`
- **Size**: 7,248 bytes (7.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: sparse"]

import warnings

from torch import nn
from torch.ao.pruning import BaseScheduler, CubicSL, LambdaSL, WeightNormSparsifier
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class ImplementedScheduler(BaseScheduler):
    def get_sl(self):
        if self.last_epoch > 0:
            return [group["sparsity_level"] * 0.5 for group in self.sparsifier.groups]
        else:
            return list(self.base_sl)


class TestScheduler(TestCase):
    def test_constructor(self):
        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        scheduler = ImplementedScheduler(sparsifier)

        assert scheduler.sparsifier is sparsifier
        assert scheduler._step_count == 1
        assert scheduler.base_sl == [sparsifier.groups[0]["sparsity_level"]]

    def test_order_of_steps(self):
        """Checks if the warning is thrown if the scheduler step is called
        before the sparsifier step"""

        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        scheduler = ImplementedScheduler(sparsifier)

        # Sparsifier step is not called
        with self.assertWarns(UserWarning):
            scheduler.step()

        # Correct order has no warnings
        # Note: This will trigger if other warnings are present.
        with warnings.catch_warnings(record=True) as w:
            sparsifier.step()
            scheduler.step()
            # Make sure there is no warning related to the base_scheduler
            for warning in w:
                fname = warning.filename
                fname = "/".join(fname.split("/")[-5:])
                assert fname != "torch/ao/sparsity/scheduler/base_scheduler.py"

    def test_step(self):
        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        assert sparsifier.groups[0]["sparsity_level"] == 0.5
        scheduler = ImplementedScheduler(sparsifier)
        assert sparsifier.groups[0]["sparsity_level"] == 0.5

        sparsifier.step()
        scheduler.step()
        assert sparsifier.groups[0]["sparsity_level"] == 0.25

    def test_lambda_scheduler(self):
        model = nn.Sequential(nn.Linear(16, 16))
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        assert sparsifier.groups[0]["sparsity_level"] == 0.5
        scheduler = LambdaSL(sparsifier, lambda epoch: epoch * 10)
        assert sparsifier.groups[0]["sparsity_level"] == 0.0  # Epoch 0
        scheduler.step()
        assert sparsifier.groups[0]["sparsity_level"] == 5.0  # Epoch 1


class TestCubicScheduler(TestCase):
    def setUp(self):
        super().setUp()
        self.model_sparse_config = [
            {"tensor_fqn": "0.weight", "sparsity_level": 0.8},
            {"tensor_fqn": "2.weight", "sparsity_level": 0.4},
        ]
        self.sorted_sparse_levels = [
            conf["sparsity_level"] for conf in self.model_sparse_config
        ]
        self.initial_sparsity = 0.1
        self.initial_step = 3

    def _make_model(self, **kwargs):
        model = nn.Sequential(
            nn.Linear(13, 17),
            nn.Dropout(0.5),
            nn.Linear(17, 3),
        )
        return model

    def _make_scheduler(self, model, **kwargs):
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=self.model_sparse_config)

        scheduler_args = {
            "init_sl": self.initial_sparsity,
            "init_t": self.initial_step,
        }
        scheduler_args.update(kwargs)

        scheduler = CubicSL(sparsifier, **scheduler_args)
        return sparsifier, scheduler

    @staticmethod
    def _get_sparsity_levels(sparsifier, precision=32):
        r"""Gets the current levels of sparsity in a sparsifier."""
        return [
            round(group["sparsity_level"], precision) for group in sparsifier.groups
        ]

    def test_constructor(self):
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(model=model, initially_zero=True)
        self.assertIs(
            scheduler.sparsifier, sparsifier, msg="Sparsifier is not properly attached"
        )
        self.assertEqual(
            scheduler._step_count,
            1,
            msg="Scheduler is initialized with incorrect step count",
        )
        self.assertEqual(
            scheduler.base_sl,
            self.sorted_sparse_levels,
            msg="Scheduler did not store the target sparsity levels correctly",
        )

        # Value before t_0 is 0
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(0.0),
            msg="Sparsifier is not reset correctly after attaching to the Scheduler",
        )

        # Value before t_0 is s_0
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(model=model, initially_zero=False)
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(self.initial_sparsity),
            msg="Sparsifier is not reset correctly after attaching to the Scheduler",
        )

    def test_step(self):
        # For n=5, dt=2, there will be totally 10 steps between s_0 and s_f, starting from t_0
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(
            model=model, initially_zero=True, init_t=3, delta_t=2, total_t=5
        )

        scheduler.step()
        scheduler.step()
        self.assertEqual(
            scheduler._step_count,
            3,
            msg="Scheduler step_count is expected to increment",
        )
        # Value before t_0 is supposed to be 0
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(0.0),
            msg="Scheduler step updating the sparsity level before t_0",
        )

        scheduler.step()  # Step = 3  =>  sparsity = initial_sparsity
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(self.initial_sparsity),
            msg="Sparsifier is not reset to initial sparsity at the first step",
        )

        scheduler.step()  # Step = 4  =>  sparsity ~ [0.3, 0.2]
        self.assertEqual(
            self._get_sparsity_levels(sparsifier, 1),
            [0.3, 0.2],
            msg="Sparsity level is not set correctly after the first step",
        )

        current_step = scheduler._step_count - scheduler.init_t[0] - 1
        more_steps_needed = scheduler.delta_t[0] * scheduler.total_t[0] - current_step
        for _ in range(more_steps_needed):  # More steps needed to final sparsity level
            scheduler.step()
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            self.sorted_sparse_levels,
            msg="Sparsity level is not reaching the target level after delta_t * n steps ",
        )


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")

```



## High-Level Overview

"""Checks if the warning is thrown if the scheduler step is called

This Python file contains 3 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ImplementedScheduler`, `TestScheduler`, `TestCubicScheduler`

**Functions defined**: `get_sl`, `test_constructor`, `test_order_of_steps`, `test_step`, `test_lambda_scheduler`, `setUp`, `_make_model`, `_make_scheduler`, `_get_sparsity_levels`, `test_constructor`, `test_step`

**Key imports**: warnings, nn, BaseScheduler, CubicSL, LambdaSL, WeightNormSparsifier, raise_on_run_directly, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/ao/sparsity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `warnings`
- `torch`: nn
- `torch.ao.pruning`: BaseScheduler, CubicSL, LambdaSL, WeightNormSparsifier
- `torch.testing._internal.common_utils`: raise_on_run_directly, TestCase


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

This is a test file. Run it with:

```bash
python test/ao/sparsity/test_scheduler.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/ao/sparsity`):

- [`test_kernels.py_docs.md`](./test_kernels.py_docs.md)
- [`test_activation_sparsifier.py_docs.md`](./test_activation_sparsifier.py_docs.md)
- [`test_data_scheduler.py_docs.md`](./test_data_scheduler.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)
- [`test_sparsity_utils.py_docs.md`](./test_sparsity_utils.py_docs.md)
- [`test_data_sparsifier.py_docs.md`](./test_data_sparsifier.py_docs.md)
- [`test_structured_sparsifier.py_docs.md`](./test_structured_sparsifier.py_docs.md)
- [`test_qlinear_packed_params.py_docs.md`](./test_qlinear_packed_params.py_docs.md)
- [`test_sparsifier.py_docs.md`](./test_sparsifier.py_docs.md)


## Cross-References

- **File Documentation**: `test_scheduler.py_docs.md`
- **Keyword Index**: `test_scheduler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/ao/sparsity`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/ao/sparsity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python docs/test/ao/sparsity/test_scheduler.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/ao/sparsity`):

- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_data_sparsifier.py_kw.md_docs.md`](./test_data_sparsifier.py_kw.md_docs.md)
- [`test_activation_sparsifier.py_docs.md_docs.md`](./test_activation_sparsifier.py_docs.md_docs.md)
- [`test_data_scheduler.py_kw.md_docs.md`](./test_data_scheduler.py_kw.md_docs.md)
- [`test_sparsity_utils.py_kw.md_docs.md`](./test_sparsity_utils.py_kw.md_docs.md)
- [`test_structured_sparsifier.py_docs.md_docs.md`](./test_structured_sparsifier.py_docs.md_docs.md)
- [`test_composability.py_kw.md_docs.md`](./test_composability.py_kw.md_docs.md)
- [`test_kernels.py_kw.md_docs.md`](./test_kernels.py_kw.md_docs.md)
- [`test_structured_sparsifier.py_kw.md_docs.md`](./test_structured_sparsifier.py_kw.md_docs.md)
- [`test_data_sparsifier.py_docs.md_docs.md`](./test_data_sparsifier.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_scheduler.py_docs.md_docs.md`
- **Keyword Index**: `test_scheduler.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
