# Documentation: `benchmarks/fastrnns/profile.py`

## File Metadata

- **Path**: `benchmarks/fastrnns/profile.py`
- **Size**: 4,586 bytes (4.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import datetime
import subprocess
import sys
import time

import torch

from .runner import get_nn_runners


def run_rnn(
    name,
    rnn_creator,
    nloops=5,
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    device="cuda",
    seed=None,
):
    def run_iter(modeldef):
        # Forward
        forward_output = modeldef.forward(*modeldef.inputs)

        # "loss computation" and backward
        if modeldef.backward_setup is not None:
            backward_input = modeldef.backward_setup(forward_output)
        else:
            backward_input = forward_output
        if modeldef.backward is not None:
            modeldef.backward(*backward_input)

        # "Update" parameters
        if modeldef.backward is not None:
            with torch.no_grad():
                for param in modeldef.params:
                    param.grad.zero_()
        torch.cuda.synchronize()

    assert device == "cuda"
    creator_args = dict(
        seqLength=seqLength,
        numLayers=numLayers,
        inputSize=inputSize,
        hiddenSize=hiddenSize,
        miniBatch=miniBatch,
        device=device,
        seed=seed,
    )
    modeldef = rnn_creator(**creator_args)

    [run_iter(modeldef) for _ in range(nloops)]


def profile(
    rnns,
    sleep_between_seconds=1,
    nloops=5,
    internal_run=True,  # Unused, get rid of this TODO
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    device="cuda",
    seed=None,
):
    params = dict(
        seqLength=seqLength,
        numLayers=numLayers,
        inputSize=inputSize,
        hiddenSize=hiddenSize,
        miniBatch=miniBatch,
        device=device,
        seed=seed,
    )
    for name, creator, context in get_nn_runners(*rnns):
        with context():
            run_rnn(name, creator, nloops, **params)
            time.sleep(sleep_between_seconds)


def system(command):
    """Returns (return-code, stdout, stderr)"""
    print(f"[system] {command}")
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = p.communicate()
    rc = p.returncode
    output = output.decode("ascii")
    err = err.decode("ascii")
    return rc, output, err


def describe_sizes(**sizes):
    # seqLength, numLayers, inputSize, hiddenSize, miniBatch
    return "s{}-l{}-i{}-h{}-b{}".format(
        sizes["seqLength"],
        sizes["numLayers"],
        sizes["inputSize"],
        sizes["hiddenSize"],
        sizes["miniBatch"],
    )


OUTPUT_DIR = "~/profout/"


def nvprof_output_filename(rnns, **params):
    rnn_tag = "-".join(rnns)
    size_tag = describe_sizes(**params)
    date_tag = datetime.datetime.now().strftime("%m%d%y-%H%M")
    return f"{OUTPUT_DIR}prof_{rnn_tag}_{size_tag}_{date_tag}.nvvp"


def nvprof(cmd, outpath):
    return system(f"nvprof -o {outpath} {cmd}")


def full_profile(rnns, **args):
    profile_args = []
    for k, v in args.items():
        profile_args.append(f"--{k}={v}")
    profile_args.append(f"--rnns {' '.join(rnns)}")
    profile_args.append("--internal-run")

    outpath = nvprof_output_filename(rnns, **args)

    cmd = f"{sys.executable} -m fastrnns.profile {' '.join(profile_args)}"
    rc, stdout, stderr = nvprof(cmd, outpath)
    if rc != 0:
        raise RuntimeError(f"stderr: {stderr}\nstdout: {stdout}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile RNNs")

    parser.add_argument("--seqLength", default="100", type=int)
    parser.add_argument("--numLayers", default="1", type=int)
    parser.add_argument("--inputSize", default="512", type=int)
    parser.add_argument("--hiddenSize", default="512", type=int)
    parser.add_argument("--miniBatch", default="64", type=int)
    parser.add_argument(
        "--sleep-between-seconds", "--sleep_between_seconds", default="1", type=int
    )
    parser.add_argument("--nloops", default="5", type=int)

    parser.add_argument("--rnns", nargs="*", help="What to run. cudnn, aten, jit, etc")

    # if internal_run, we actually run the rnns.
    # if not internal_run, we shell out to nvprof with internal_run=T
    parser.add_argument(
        "--internal-run",
        "--internal_run",
        default=False,
        action="store_true",
        help="Don't use this",
    )
    args = parser.parse_args()
    if args.rnns is None:
        args.rnns = ["cudnn", "aten", "jit"]
    print(args)

    if args.internal_run:
        profile(**vars(args))
    else:
        full_profile(**vars(args))

```



## High-Level Overview


This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `run_rnn`, `run_iter`, `profile`, `system`, `describe_sizes`, `nvprof_output_filename`, `nvprof`, `full_profile`

**Key imports**: argparse, datetime, subprocess, sys, time, torch, get_nn_runners


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/fastrnns`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `datetime`
- `subprocess`
- `sys`
- `time`
- `torch`
- `.runner`: get_nn_runners


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`benchmarks/fastrnns`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test.py_docs.md`](./test.py_docs.md)
- [`cells.py_docs.md`](./cells.py_docs.md)
- [`bench.py_docs.md`](./bench.py_docs.md)
- [`conftest.py_docs.md`](./conftest.py_docs.md)
- [`custom_lstms.py_docs.md`](./custom_lstms.py_docs.md)
- [`factory.py_docs.md`](./factory.py_docs.md)
- [`fuser.py_docs.md`](./fuser.py_docs.md)
- [`runner.py_docs.md`](./runner.py_docs.md)


## Cross-References

- **File Documentation**: `profile.py_docs.md`
- **Keyword Index**: `profile.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
