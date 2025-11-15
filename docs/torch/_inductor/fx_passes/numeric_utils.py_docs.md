# Documentation: `torch/_inductor/fx_passes/numeric_utils.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/numeric_utils.py`
- **Size**: 7,285 bytes (7.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import gc
import logging
import os
import random
import traceback

import numpy

import torch
import torch.optim as optim
from torch.utils._ordered_set import OrderedSet

from .. import config


logger: logging.Logger = logging.getLogger(__name__)

MAIN_RANDOM_SEED = 1337

# Set the CUBLAS_WORKSPACE_CONFIG environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# If the two forward functions involve any non-deterministic operations,
# such as certain types of parallelism or asynchronous execution,
# this can also lead to different outputs.
def set_deterministic() -> None:
    """Make torch manual seed deterministic."""

    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)
    torch.use_deterministic_algorithms(True)


def clean_memory() -> None:
    """Clean memory to avoid OOM."""
    gc.collect()
    torch.cuda.empty_cache()


# We compare the numerical results before and after pre/post grad fx passes
# transformation to make sure the numerical results are the same.
def compare_dict_tensors(dict_base, dict_control, precision):
    if len(OrderedSet(dict_base.keys())) != len(OrderedSet(dict_control.keys())):
        logger.warning("Mismatch keys found before and after pre/post grad fx passes.")
        logger.debug("keys before pre/post grad fx passes %s", dict_base.keys())
        logger.debug("keys after pre/post grad fx passes %s", dict_control.keys())
        return False
    is_allclose = True
    for key in dict_base:
        if key not in dict_control:
            logger.warning(
                "Mismatch parameter name %s does not exist after pre/post grad fx passes",
                key,
            )
        # Some parameters have `None`, and not every param has a valid .grad field, we skip them
        if dict_base[key] is None or dict_control[key] is None:
            continue
        if not torch.allclose(
            dict_base[key],
            dict_control[key],
            rtol=precision,
            atol=precision,
            equal_nan=True,
        ):
            logger.warning(
                "Mismatch parameter values found before and after pre/post grad fx passes."
            )
            logger.debug("value before pre/post grad fx passes %s", dict_base[key])
            logger.debug("value after pre/post grad fx passes %s", dict_control[key])
            is_allclose = False
    return is_allclose


def compare_tuple_tensors(tuple_base, tuple_control, precision):
    if len(tuple_base) != len(tuple_control):
        logger.warning(
            "Mismatch fw output length. before transformation: %s, after transformation: %s",
            len(tuple_base),
            len(tuple_control),
        )
        return False
    is_allclose = True
    for i in range(len(tuple_base)):
        # Some parameters have `None`, we skip them
        if tuple_base[i] is None or tuple_control[i] is None:
            continue
        if not torch.allclose(
            tuple_base[i],
            tuple_control[i],
            rtol=precision,
            atol=precision,
            equal_nan=True,
        ):
            logger.debug(
                "forward output before pre/post grad fx passes %s", tuple_base[i]
            )
            logger.debug(
                "forward output after pre/post grad fx passes %s", tuple_control[i]
            )
            is_allclose = False
    return is_allclose


def compare_parameters(model_base, model_control, precision):
    return compare_dict_tensors(
        dict(model_base.named_parameters()),
        dict(model_control.named_parameters()),
        precision,
    )


def compare_forward_output(pred_base, pred_control, precision):
    return compare_tuple_tensors(
        pred_base,
        pred_control,
        precision,
    )


def compare_gradients(model_base, model_control, precision):
    grad_base = {key: param.grad for key, param in model_base.named_parameters()}
    grad_pt2 = {key: param.grad for key, param in model_control.named_parameters()}
    return compare_dict_tensors(
        grad_base,
        grad_pt2,
        precision,
    )


def run_model(
    model_base, model_control, model_input, num_iterations=10, precision=1e-4
):
    clean_memory()
    for i in range(num_iterations):
        logger.info("start %s iteration", i)
        set_deterministic()
        pred_base = model_base(*model_input)
        set_deterministic()
        pred_control = model_control(*model_input)

        res = compare_parameters(model_base, model_control, precision)
        logger.info("compare parameters. Numerical result : %s", res)

        res = compare_forward_output(pred_base, pred_control, precision)
        logger.info("compare loss/predict. Numerical result : %s", res)
        # tensor may not have a grad_fn
        try:
            _ = pred_base[0].sum().backward(retain_graph=True)
            _ = pred_control[0].sum().backward(retain_graph=True)
            res = compare_gradients(model_base, model_control, precision)
            logger.info("compare param grad. Numerical result : %s", res)
        except Exception:
            logger.exception("Exception when comparing gradients")
            traceback.print_exc()

        if config.fx_passes_numeric_check["requires_optimizer"]:
            try:
                optimizer_base = optim.SGD(
                    [param for name, param in model_base.named_parameters()], lr=0.01
                )
                optimizer_base.step()

                optimizer_control = optim.SGD(
                    [param for name, param in model_control.named_parameters()], lr=0.01
                )
                optimizer_control.step()

                res = compare_parameters(model_base, model_control, precision)
                logger.info(
                    "compare parameters with optimizer added. Numerical result : %s",
                    res,
                )
            except Exception:
                logger.exception(
                    "Exception when optimizer is added to check parameter names"
                )
                traceback.print_exc()
        else:
            logger.warning(
                "no parameter with optimizer to compare with length %s before transformation"
                " and the length %s after transformation",
                len(dict(model_base.named_parameters())),
                len(dict(model_control.named_parameters())),
            )


def numeric_check_if_enabled(
    gm_before_fx_passes,
    gm_after_fx_passes,
    example_inputs,
    num_iterations,
    precision,
):
    # need to topo-sort graphmodule before we run the model,
    # otherwise it may fail as refer before def
    # fail silently in order not to block the model run
    try:
        with torch.autograd.set_detect_anomaly(True):
            run_model(
                gm_before_fx_passes,
                gm_after_fx_passes,
                example_inputs,
                num_iterations=num_iterations,
                precision=precision,
            )
    except Exception as e:
        logger.warning(  # noqa: G200
            "Runtime numeric check failed in pre grad fx passes with error: %s", e
        )
        traceback.print_exc()

```



## High-Level Overview

"""Make torch manual seed deterministic."""    torch.manual_seed(MAIN_RANDOM_SEED)    random.seed(MAIN_RANDOM_SEED)    numpy.random.seed(MAIN_RANDOM_SEED)    torch.use_deterministic_algorithms(True)def clean_memory() -> None:

This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `set_deterministic`, `clean_memory`, `compare_dict_tensors`, `compare_tuple_tensors`, `compare_parameters`, `compare_forward_output`, `compare_gradients`, `run_model`, `numeric_check_if_enabled`

**Key imports**: gc, logging, os, random, traceback, numpy, torch, torch.optim as optim, OrderedSet, config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `gc`
- `logging`
- `os`
- `random`
- `traceback`
- `numpy`
- `torch`
- `torch.optim as optim`
- `torch.utils._ordered_set`: OrderedSet
- `..`: config


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/_inductor/fx_passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fuse_attention.py_docs.md`](./fuse_attention.py_docs.md)
- [`efficient_conv_bn_eval.py_docs.md`](./efficient_conv_bn_eval.py_docs.md)
- [`bucketing.py_docs.md`](./bucketing.py_docs.md)
- [`dedupe_symint_uses.py_docs.md`](./dedupe_symint_uses.py_docs.md)
- [`post_grad.py_docs.md`](./post_grad.py_docs.md)
- [`joint_graph.py_docs.md`](./joint_graph.py_docs.md)
- [`fsdp.py_docs.md`](./fsdp.py_docs.md)


## Cross-References

- **File Documentation**: `numeric_utils.py_docs.md`
- **Keyword Index**: `numeric_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
