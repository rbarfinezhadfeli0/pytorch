# Documentation: `docs/torch/distributed/launch.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/launch.py_docs.md`
- **Size**: 12,218 bytes (11.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/launch.py`

## File Metadata

- **Path**: `torch/distributed/launch.py`
- **Size**: 7,595 bytes (7.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
r"""
Module ``torch.distributed.launch``.

``torch.distributed.launch`` is a module that spawns up multiple distributed
training processes on each of the training nodes.

.. warning::

    This module is going to be deprecated in favor of :ref:`torchrun <launcher-api>`.

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be beneficial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed
training, this utility will launch the given number of processes per node
(``--nproc-per-node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

**How to use this module:**

1. Single-Node multi-process distributed training

::

    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

2. Multi-Node multi-process distributed training: (e.g. two nodes)


Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node-rank=0 --master-addr="192.168.1.1"
               --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

Node 2:

::

    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node-rank=1 --master-addr="192.168.1.1"
               --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

3. To look up what optional arguments this module offers:

::

    python -m torch.distributed.launch --help


**Important Notices:**

1. This utility and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

2. In your training program, you must parse the command-line argument:
``--local-rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:

Parsing the local_rank argument

::

    >>> # xdoctest: +SKIP
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local-rank", "--local_rank", type=int)
    >>> args = parser.parse_args()

Set your device to local rank using either

::

    >>> torch.cuda.set_device(args.local_rank)  # before your code runs

or

::

    >>> with torch.cuda.device(args.local_rank):
    >>>    # your code to run
    >>>    ...

.. versionchanged:: 2.0.0

    The launcher will passes the ``--local-rank=<rank>`` argument to your script.
    From PyTorch 2.0.0 onwards, the dashed ``--local-rank`` is preferred over the
    previously used underscored ``--local_rank``.

    For backward compatibility, it may be necessary for users to handle both
    cases in their argument parsing code. This means including both ``"--local-rank"``
    and ``"--local_rank"`` in the argument parser. If only ``"--local_rank"`` is
    provided, the launcher will trigger an error: "error: unrecognized arguments:
    --local-rank=<rank>". For training code that only supports PyTorch 2.0.0+,
    including ``"--local-rank"`` should be sufficient.

3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. It is strongly recommended
that ``init_method=env://``. Other init methods (e.g. ``tcp://``) may work,
but ``env://`` is the one that is officially supported by this module.

::

    >>> torch.distributed.init_process_group(backend='YOUR BACKEND',
    >>>                                      init_method='env://')

4. In your training program, you can either use regular distributed functions
or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
training program uses GPUs for training and you would like to use
:func:`torch.nn.parallel.DistributedDataParallel` module,
here is how to configure it.

::

    >>> model = torch.nn.parallel.DistributedDataParallel(model,
    >>>                                                   device_ids=[args.local_rank],
    >>>                                                   output_device=args.local_rank)

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[args.local_rank]``,
and ``output_device`` needs to be ``args.local_rank`` in order to use this
utility

5. Another way to pass ``local_rank`` to the subprocesses via environment variable
``LOCAL_RANK``. This behavior is enabled when you launch the script with
``--use-env=True``. You must adjust the subprocess example above to replace
``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local-rank`` when you specify this flag.

.. warning::

    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.



"""

from typing_extensions import deprecated as _deprecated

from torch.distributed.run import get_args_parser, run


def parse_args(args):
    parser = get_args_parser()
    parser.add_argument(
        "--use-env",
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local-rank as argument, and will instead set LOCAL_RANK.",
    )
    return parser.parse_args(args)


def launch(args):
    if args.no_python and not args.use_env:
        raise ValueError(
            "When using the '--no-python' flag, you must also set the '--use-env' flag."
        )
    run(args)


@_deprecated(
    "The module torch.distributed.launch is deprecated\n"
    "and will be removed in future. Use torchrun.\n"
    "Note that --use-env is set by default in torchrun.\n"
    "If your script expects `--local-rank` argument to be set, please\n"
    "change it to read from `os.environ['LOCAL_RANK']` instead. See \n"
    "https://pytorch.org/docs/stable/distributed.html#launch-utility for \n"
    "further instructions\n",
    category=FutureWarning,
)
def main(args=None):
    args = parse_args(args)
    launch(args)


if __name__ == "__main__":
    main()

```



## High-Level Overview

r"""Module ``torch.distributed.launch``.``torch.distributed.launch`` is a module that spawns up multiple distributedtraining processes on each of the training nodes... warning::    This module is going to be deprecated in favor of :ref:`torchrun <launcher-api>`.The utility can be used for single-node distributed training, in which one ormore processes per node will be spawned. The utility can be used for eitherCPU training or GPU training. If the utility is used for GPU training,each distributed process will be operating on a single GPU. This can achievewell-improved single-node training performance. It can also be used inmulti-node distributed training, by spawning up multiple processes on each nodefor well-improved multi-node distributed training performance as well.This will especially be beneficial for systems with multiple Infinibandinterfaces that have direct-GPU support, since all of them can be utilized foraggregated communication bandwidth.In both cases of single-node distributed training or multi-node distributedtraining, this utility will launch the given number of processes per node(``--nproc-per-node``). If used for GPU training, this number needs to be lessor equal to the number of GPUs on the current system (``nproc_per_node``),and each process will be operating on a single GPU from *GPU 0 toGPU (nproc_per_node - 1)*.**How to use this module:**1. Single-Node multi-process distributed training::    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other               arguments of your training script)2. Multi-Node multi-process distributed training: (e.g. two nodes)Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*::    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE               --nnodes=2 --node-rank=0 --master-addr="192.168.1.1"               --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3               and all other arguments of your training script)

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `parse_args`, `launch`, `main`

**Key imports**: argparse, deprecated as _deprecated, get_args_parser, run


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `typing_extensions`: deprecated as _deprecated
- `torch.distributed.run`: get_args_parser, run


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_mesh_layout.py_docs.md`](./_mesh_layout.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`c10d_logger.py_docs.md`](./c10d_logger.py_docs.md)
- [`_functional_collectives.py_docs.md`](./_functional_collectives.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`CONTRIBUTING.md_docs.md`](./CONTRIBUTING.md_docs.md)
- [`_functional_collectives_impl.py_docs.md`](./_functional_collectives_impl.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)


## Cross-References

- **File Documentation**: `launch.py_docs.md`
- **Keyword Index**: `launch.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed`):

- [`_mesh_layout.py_docs.md_docs.md`](./_mesh_layout.py_docs.md_docs.md)
- [`run.py_docs.md_docs.md`](./run.py_docs.md_docs.md)
- [`device_mesh.py_docs.md_docs.md`](./device_mesh.py_docs.md_docs.md)
- [`_composable_state.py_docs.md_docs.md`](./_composable_state.py_docs.md_docs.md)
- [`run.py_kw.md_docs.md`](./run.py_kw.md_docs.md)
- [`_dist2.py_kw.md_docs.md`](./_dist2.py_kw.md_docs.md)
- [`_state_dict_utils.py_kw.md_docs.md`](./_state_dict_utils.py_kw.md_docs.md)
- [`rendezvous.py_kw.md_docs.md`](./rendezvous.py_kw.md_docs.md)
- [`rendezvous.py_docs.md_docs.md`](./rendezvous.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `launch.py_docs.md_docs.md`
- **Keyword Index**: `launch.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
