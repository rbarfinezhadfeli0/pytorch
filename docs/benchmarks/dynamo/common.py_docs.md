# Documentation: `benchmarks/dynamo/common.py`

## File Metadata

- **Path**: `benchmarks/dynamo/common.py`
- **Size**: 160,584 bytes (156.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import csv
import dataclasses
import functools
import gc
import importlib
import itertools
import json
import logging
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import weakref
from contextlib import contextmanager
from typing import Any, NamedTuple, Optional, overload, TYPE_CHECKING, TypeVar
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import psutil
import yaml
from scipy.stats import gmean, ttest_ind
from tqdm.auto import tqdm, trange

import torch
import torch._dynamo
import torch._dynamo.utils
import torch._export
import torch.distributed
import torch.multiprocessing as mp
from torch._C import _has_cuda as HAS_CUDA, _has_xpu as HAS_XPU
from torch._C._nativert import PyModelRunner
from torch._dynamo.profiler import fx_insert_profiling, Profiler
from torch._dynamo.testing import (
    dummy_fx_compile,
    format_speedup,
    reset_rng_state,
    same,
)
from torch._dynamo.utils import bitwise_same
from torch._logging.scribe import open_source_signpost


try:
    from torch._dynamo.utils import clone_inputs, graph_break_reasons
    from torch._inductor.utils import fresh_cache
except ImportError:
    from _dynamo.utils import clone_inputs, graph_break_reasons
    from _inductor.utils import fresh_cache

import torch._functorch.config
from torch._functorch.aot_autograd import set_model_name
from torch._inductor import config as inductor_config, metrics
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map, tree_map_only


try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    # This is to workaround the backward issue https://github.com/pytorch/xla/issues/4174
    torch_xla._XLAC._init_computation_client()
except ImportError:
    # ignore the error if torch_xla is not installed
    pass


if TYPE_CHECKING:
    from collections.abc import Sequence

_D = TypeVar("_D", bound=dict[str, Any])
_T = TypeVar("_T")


log = logging.getLogger(__name__)

# We are primarily interested in TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

# Suppress torch.profiler spam
os.environ["KINETO_LOG_LEVEL"] = "5"

current_name = ""
current_device = ""
current_backend = ""
current_mode = ""
current_dtype = ""
current_quantization = ""
current_settings = None
current_batch_size = None
output_filename = None
disable_output = False

MAX_DOWNLOAD_ATTEMPTS = 5


class CI(NamedTuple):
    backend: str  # aot_eager or inductor
    training: bool
    dynamic: bool = False
    device: str = "cuda"


CI_SKIP_OPTIMIZER = {
    # HF
    "MobileBertForMaskedLM",  # Stack issue in fx
}

try:
    from .fb.common import INTERNAL_CI_SKIP_DYNAMIC_BATCH_ONLY
except ImportError:
    INTERNAL_CI_SKIP_DYNAMIC_BATCH_ONLY = set()

try:
    from pytorch.benchmark.fb.run_utils import trace_handler
except ImportError:
    trace_handler = None


CI_SKIP_DYNAMIC_BATCH_ONLY = {
    "sam",
    # See https://github.com/mindee/doctr/blob/f2114758d529ed8d3d0030581638f0520b6b98d8/doctr/models/detection/core.py#L89
    # It iterates over the batch, which is dynamic, and dynamo chokes
    # We should be able to graphbreak there.
    "doctr_det_predictor",
    "dlrm",
    "pyhpc_isoneutral_mixing",
    "pyhpc_equation_of_state",
    "pyhpc_turbulent_kinetic_energy",
    "detectron2_fcos_r_50_fpn",
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "Reformer",
    "llama",
}.union(INTERNAL_CI_SKIP_DYNAMIC_BATCH_ONLY)

# These models currently fail accuracy with eager Adam optimizer
# so we use SGD when running the full benchmarks
# https://github.com/pytorch/pytorch/issues/115966
BENCHMARK_USE_SGD = {
    # TorchBench
    "BERT_pytorch",
    "LearningToPaint",
    "alexnet",
    "dcgan",
    "demucs",
    "densenet121",
    "dlrm",
    "fastNLP_Bert",
    "mobilenet_v2",
    "phlippe_densenet",
    "phlippe_resnet",
    "pytorch_stargan",
    "resnet18",
    "shufflenet_v2_x1_0",
    "speech_transformer",
    "squeezenet1_1",
    "stable_diffusion_text_encoder",
    "vgg16",
    # HF
    "AlbertForMaskedLM",
    "BartForCausalLM",
    "ElectraForCausalLM",
    "M2M100ForConditionalGeneration",
    "MBartForCausalLM",
    "OPTForCausalLM",
    "PLBartForCausalLM",
    "PegasusForCausalLM",
    "TrOCRForCausalLM",
    "XGLMForCausalLM",
    # TIMM
    "adv_inception_v3",
    "tf_efficientnet_b0",
    "ghostnet_100",
}

# These models OOM in CI
# due to the extra memory of Adam optimizer states,
# so we fall back to SGD in CI
CI_USE_SGD = {
    "torchrec_dlrm",
    "demucs",
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_maskrcnn_r_101_c4",
    "detectron2_maskrcnn_r_101_fpn",
    "detectron2_maskrcnn_r_50_c4",
    "detectron2_maskrcnn_r_50_fpn",
    "llama_v2_7b_16h",
    "mobilenet_v2_quantized_qat",
    "phi_1_5 resnet50_quantized_qat",
    "BlenderbotForCausalLM",
    "DALLE2_pytorch",
    "moco",
    "timm_efficientdet",
    "ghostnet_100",
    "inception_v3",
    "mobilevit_s",
    "pytorch_CycleGAN_and_pix2pix",
    "vision_maskrcnn",
    "dlrm",
    "resnet50",
    "dm_nfnet_f0",
}


DO_NOT_CAST_INPUTS = {"stable_diffusion"}


# Maps a benchmark model name to a list of status codes. For any listed entry, we'll
# capture TORCH_COMPILE_DEBUG logs in CI runs and preserve them (i.e., for upload) if
# the result status matches one listed.
CI_PRESERVE_COMPILE_DEBUG = {
    # For example:
    # "mnasnet1_0": ["fail_accuracy"],
}


@functools.lru_cache(maxsize=1)
def load_yaml_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)

    with open(filepath) as f:
        data = yaml.safe_load(f)

    internal_file_path = os.path.join(os.path.dirname(__file__), "fb", filename)
    if os.path.exists(internal_file_path):
        with open(internal_file_path) as f:
            internal_data = yaml.safe_load(f)
            data.update(internal_data)

    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    def maybe_list_to_set(obj):
        if isinstance(obj, dict):
            return {k: maybe_list_to_set(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return set(flatten(obj))
        return obj

    return maybe_list_to_set(data)


def model_specified_by_path(path_and_class_str):
    return ":" in path_and_class_str


def load_model_from_path(path_and_class_str):
    configs = {}
    for kvstr in path_and_class_str.split(","):
        k, v = kvstr.split(":")
        configs[k] = v

    for name in ["path", "class"]:
        if name not in configs:
            raise RuntimeError(
                "Invalid --only arguments. Check help message for the correct format"
            )

    path = configs["path"]
    class_name = configs["class"]

    if path[:1] != "/":
        raise RuntimeError(
            "Use absolute path since dynamo may change the current working directory which makes using relative path tricky"
        )

    spec = importlib.util.spec_from_file_location("module_name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, class_name)
    assert issubclass(model_class, torch.nn.Module)
    model = model_class()
    assert hasattr(model, "get_example_inputs")
    inputs = model.get_example_inputs()
    return model, inputs


def write_outputs(filename, headers, row, upload_to_benchmark_db: bool = True):
    """
    Write both CSV and JSON outputs using the original CSV output interface
    """
    global disable_output
    if disable_output:
        return

    output_csv(filename, headers, row)
    if upload_to_benchmark_db:
        output_json(filename, headers, row)


def output_csv(filename, headers, row):
    if os.path.exists(filename):
        with open(filename) as fd:
            lines = list(csv.reader(fd)) or [[]]
            if headers and len(headers) > len(lines[0]):
                # if prior results failed the header might not be filled in yet
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    with open(filename, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


def output_json(filename, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    origin = ""
    if "torchbench" in filename:
        origin = "torchbench"
    elif "huggingface" in filename:
        origin = "huggingface"
    elif "timm_models" in filename:
        origin = "timm_models"

    extra_info = {
        "device": current_device,
        "quantization": current_quantization,
        "batch_size": current_batch_size,
    }
    if current_settings:
        extra_info.update(current_settings)

    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    with open(f"{os.path.splitext(filename)[0]}.json", "a") as f:
        for header, value in mapping_headers.items():
            # These headers are not metric names
            if header in ("dev", "name", "batch_size"):
                continue

            # Make sure that the record is valid
            if not current_name:
                continue

            record = {
                "benchmark": {
                    "name": "TorchInductor",
                    "mode": current_mode,
                    "dtype": current_dtype,
                    "extra_info": extra_info,
                },
                "model": {
                    "name": current_name,
                    "type": "OSS model",
                    "backend": current_backend,
                    "origins": [origin],
                },
            }

            # NB: When the metric is accuracy, its value is actually a string, i.e. pass, and
            # not a number. ClickHouse doesn't support mix types atm. It has a Variant type
            # https://clickhouse.com/docs/en/sql-reference/data-types/variant, but this isn't
            # recommended by CH team themselves. The workaround here is to store that value
            # in the extra_info field instead.
            if isinstance(value, str):
                record["metric"] = {
                    "name": header,
                    "extra_info": {"benchmark_values": [value]},
                }
            else:
                record["metric"] = {
                    "name": header,
                    "benchmark_values": [value],
                }

            print(json.dumps(record), file=f)


def get_suite_from_model_iter_fn(model_iter_fn):
    # TODO: This is a bit of a hack
    suite = None
    if (runner := getattr(model_iter_fn, "__self__", None)) and hasattr(
        runner, "suite_name"
    ):
        suite = runner.suite_name
    return suite


def output_signpost(data, args, suite, error=None):
    from torch.utils._stats import simple_call_counter

    data = data.copy()

    if "name" not in data:
        data["name"] = current_name

    if "dev" not in data:
        data["dev"] = current_device

    filtered_args = vars(args).copy()
    # I generated this list by reading through all the configs and dropping
    # ones that looked irrelevant or redundant
    for k in [
        "filter",
        "exclude",
        "exclude_exact",
        "dump_raw_metrics",
        "log_operator_inputs",
        "distributed_master_port",
        "skip_accuracy_check",
        "generate_aot_autograd_stats",
        "output",
        "output_directory",
        "disable_output",
        "export_profiler_trace",
        "profiler_trace_name",
        "explain",
        "stats",
        "print_memory",
        "print_compilation_time",
        "print_dataframe_summary",
        "print_graph_breaks",
        "log_graph_breaks",
        "timing",
        "progress",
        "timeout",
        "per_process_memory_fraction",
        "minify",
        "verbose",
        "quiet",
        "print_fx",
        "print_aten_ops",
        "log_conv_args",
        "recompile_profiler",
        "find_batch_sizes",
        # Redundant
        "batch_size",
        "batch_size_file",
        "only",
        "diff_branch",
        "tag",
        "coverage",
        "overhead",
        "speedup_dynamo_ts",
        "speedup_fx2trt",
        "speedup_fx2trt_fp16",
        "accuracy",
        "performance",
        "tolerance",
    ]:
        del filtered_args[k]

    event_name = "unknown"
    if args.accuracy:
        event_name = "accuracy"
    elif args.quantization:
        event_name = "quantization"
    elif args.performance:
        event_name = "performance"

    from torch._dynamo.utils import calculate_time_spent, compilation_time_metrics

    wall_time_by_phase = calculate_time_spent()

    open_source_signpost(
        subsystem="dynamo_benchmark",
        name=event_name,
        parameters=json.dumps(
            {
                **data,
                # TODO: Arguably the rest of these should be in the CSV too
                "suite": suite,
                # Better than using compile_times utils directly
                # NB: Externally, compilation_metrics colloquially refers to
                # the coarse-grained phase timings, even though internally
                # they are called something else
                "compilation_metrics": wall_time_by_phase,
                "agg_compilation_metrics": {
                    k: sum(v) for k, v in compilation_time_metrics.items()
                },
                "detailed_compilation_metrics": compilation_time_metrics,
                "simple_call_counter": simple_call_counter,
                # NB: args has training vs inference
                "args": filtered_args,
                "error": error,
            }
        ),
    )

    return wall_time_by_phase["total_wall_time"]


def nothing(f):
    return f


@functools.cache
def patch_torch_manual_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        seed = 1337
        if HAS_CUDA:
            import torch.cuda

            if not torch.cuda._is_in_bad_fork():
                torch.cuda.manual_seed_all(seed)
        if HAS_XPU:
            import torch.xpu

            if not torch.xpu._is_in_bad_fork():
                torch.xpu.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed = deterministic_torch_manual_seed


def empty_gpu_cache(device):
    """
    Explicitly empty gpu cache to avoid OOM in subsequent run.
    """

    if device not in ["cuda", "xpu", "mps"]:
        log.warning(
            "Trying to call the empty_gpu_cache for device: %s, which is not in list [cuda, xpu]",
            device,
        )
        return

    getattr(torch, device).empty_cache()


def synchronize():
    pass


def summarize_graph_break(filename):
    """
    Sorts and de-dupes the graphs breaks on the reason string. Note that this
    function is just a best effort to reduce the logging information. We could
    miss some graph breaks because of de-duping. We can further refine this
    function as need arises.
    """
    log_file = f"{filename.rstrip('.csv')}_graph_breaks.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = df.sort_values("reason").drop_duplicates(subset="reason")

        # Specialize for multi tensor sgd as reason is not identical
        multi_tensor_sgd_row = df.loc[df["reason"].str.contains("_multi_tensor_sgd")]
        if len(multi_tensor_sgd_row):
            df = df[
                ~df["reason"].str.contains("_multi_tensor_sgd")
            ]  # Drop all sgd rows
            df = pd.concat(
                [df, pd.DataFrame([multi_tensor_sgd_row.iloc[0]])], axis=0
            )  # Add back a single row
        df.to_csv(f"{log_file.rstrip('.csv')}_deduped.csv", index=False)


def print_summary(filename, print_dataframe=False):
    if not (filename and os.path.exists(filename)):
        return
    data = pd.read_csv(filename)
    if "tag" in data.columns:
        for tag in data.tag.unique():
            if tag == "0.0000":
                continue  # This happens for failed runs
            print(f"\nSummary for tag={tag}:")
            print_summary_table(data[data.tag == tag], print_dataframe=print_dataframe)
    else:
        print_summary_table(data, print_dataframe=print_dataframe)
    summarize_graph_break(filename)


def print_summary_table(data, print_dataframe=False):
    if print_dataframe:
        pd.options.display.max_rows = 1000
        pd.options.display.max_columns = 1000
        pd.options.display.width = 2000
        print(data)
    width = max(map(len, data.columns))
    for col in data.columns:
        try:
            if col in ("dev", "name", "batch_size", "tag"):
                continue
            elif col in ("pct_ops", "pct_time"):
                print(col.ljust(width), f"{data[col].mean():.3%}")
            elif col in ("graphs", "graph_calls", "captured_ops", "total_ops"):
                print(col.ljust(width), f"{data[col].mean():.3f}")
            elif col in ("compilation_latency"):
                print(col.ljust(width), f"mean={data[col].mean():.3f} seconds")
            elif col in ("compression_ratio"):
                print(col.ljust(width), f"mean={data[col].mean():.3f}x")
            elif col in ("accuracy"):
                pass_rate = (data[col] == "pass").mean()
                print(col.ljust(width), f"pass_rate={100 * pass_rate:.2f}%")
            else:
                cdata = data[col]
                print(
                    col.ljust(width),
                    f"gmean={gmean(cdata):.2f}x mean={cdata.mean():.3f}x",
                )
        except Exception:
            pass


def tensor_is_on_xla(tensors):
    def visit(x: torch.Tensor):
        nonlocal result
        if x.device.type == "xla":
            result = True

    result = False
    tree_map_only(torch.Tensor, visit, tensors)
    return result


def timed(
    model,
    model_iter_fn,
    example_inputs,
    times=1,
    return_result=False,
    collect_outputs=False,
    batch_size=None,
):
    use_xla = tensor_is_on_xla(example_inputs)
    synchronize()

    if batch_size:
        patch_torch_manual_seed()

    if use_xla:
        xm.mark_step()
        xm.wait_device_ops()

    def vary_batch(t: torch.Tensor, new_batch_size) -> torch.Tensor:
        for i, s in enumerate(t.size()):
            if s == batch_size:
                # If new batch is smaller, we truncate
                if new_batch_size < batch_size:
                    indexer = [slice(None)] * t.ndim
                    indexer[i] = slice(0, new_batch_size)
                    t = t[tuple(indexer)]
                # If new batch is greater, we just duplicate the last row
                # over and over until we hit the desired batch size
                elif new_batch_size > batch_size:
                    indexer = [slice(None)] * t.ndim
                    indexer[i] = -1
                    last_slice = t[tuple(indexer)].unsqueeze(i)
                    repeat_shape = list(t.shape)
                    repeat_shape[i] = new_batch_size - batch_size
                    padding = last_slice.expand(*repeat_shape)
                    t = torch.cat([t, padding], dim=i)
                break
        return t

    time_total = 0
    # Dont collect outputs to correctly measure timing
    for i in range(times):
        # If batch_size is 1, it too often collides with other non batch size
        # dimensions resulting in errors.
        if batch_size and batch_size > 1:
            # Calculate new batch size by varying the original batch size by up to 20%
            # Ensure it's at least greater than 1
            variation = random.uniform(0.8, 1.2)
            new_batch_size = max(2, int(batch_size * variation))
            example_inputs = tree_map_only(
                torch.Tensor, lambda x: vary_batch(x, new_batch_size), example_inputs
            )
        # Put this call inside the loop to reset the seed for each iteration.
        # Don't include reset_rng_state() to correctly measure timing
        reset_rng_state(use_xla)
        t_iter_begin = time.perf_counter()
        result = model_iter_fn(model, example_inputs, collect_outputs=collect_outputs)

        # instead of calling sync on result_list, we should call mark_step.
        # In training case, result_list may be empty, but we want to
        # send all the pending graphs for compilation.
        if use_xla:
            # For the model running on regular torchxla (baseline), we need the
            # mark step to send the accumulated graph for compilation.
            #
            # For the model running with dynamo/torchxla bridge, in training case,
            # we need the mark step to send the optimizer graph out for
            # compilation.
            xm.mark_step()
        t_iter_end = time.perf_counter()
        time_total += t_iter_end - t_iter_begin

    t_0 = time.perf_counter()
    if use_xla:
        xm.wait_device_ops()
    synchronize()
    t_1 = time.perf_counter()
    time_total += t_1 - t_0
    return (time_total, result) if return_result else time_total


@overload
def _normalize_bench_inputs(example_inputs: _D) -> tuple[tuple[()], _D]: ...


@overload
def _normalize_bench_inputs(
    example_inputs: Sequence[_T],
) -> tuple[tuple[_T, ...], dict[str, Any]]: ...


def _normalize_bench_inputs(example_inputs):
    # NOTE(bowbao): For huggingface benchmark, example_inputs are formatted as dictionary,
    # and consumed like `model(**example_inputs)`.
    # For other benchmarks, example_inputs are formatted as tuple and consumed
    # like `model(*example_inputs)`.
    if isinstance(example_inputs, dict):
        return (), example_inputs
    else:
        return tuple(example_inputs), {}


def _register_dataclass_output_as_pytree(example_outputs) -> None:
    # NOTE(angelayi): For huggingface benchmark, some example outputs are
    # formatted as a dataclass which pytree cannot consume. So we want
    # to register the pytree implementation here
    example_outputs_flat = pytree.tree_leaves(example_outputs)
    output_dataclass_types = [
        type(out) for out in example_outputs_flat if dataclasses.is_dataclass(type(out))
    ]
    for output_type in output_dataclass_types:
        from torch._export.utils import register_dataclass_as_pytree_node

        register_dataclass_as_pytree_node(
            output_type,
            serialized_type_name=f"{output_type.__module__}.{output_type.__name__}",
        )


class Stats:
    totals = collections.defaultdict(collections.Counter)

    @classmethod
    def reset_counters(cls):
        for k, v in torch._dynamo.utils.counters.items():
            cls.totals[k].update(v)
        ok = torch._dynamo.utils.counters["frames"]["ok"]
        total = torch._dynamo.utils.counters["frames"]["total"]
        torch._dynamo.utils.counters.clear()
        return ok, total

    @classmethod
    def print_summary(cls):
        for k, v in sorted(cls.totals.items()):
            lines = "\n  ".join(map(str, v.most_common(50)))
            print(f"STATS {k}\n  {lines}")

    @classmethod
    def aot_summary(cls):
        return [cls.totals["aot_autograd"]["total"], cls.totals["aot_autograd"]["ok"]]


def coverage_experiment(args, model_iter_fn, model, example_inputs):
    """
    Test operator/model coverage of TorchDynamo and record statistics
    taken from a profiler.  This target is mainly intended to check
    correctness.

    Writes to ./coverage.csv
    """
    profiler = Profiler()
    frozen_model_iter_fn = torch._dynamo.run(model_iter_fn)
    with profiler.prof:
        frozen_model_iter_fn(model, example_inputs)
    coverage_result = profiler.results()
    write_outputs(
        output_filename,
        (
            "dev",
            "name",
            "batch_size",
            "graphs",
            "graph_calls",
            "captured_ops",
            "total_ops",
            "pct_ops",
            "pct_time",
        ),
        [
            current_device,
            current_name,
            current_batch_size,
        ]
        + coverage_result.tocsv(),
    )
    return coverage_result


def speedup_experiment_fx2trt(args, model_iter_fn, model, example_inputs):
    """
    Measure speedups over eager using the trt inference backend. TRT backend is based fx graph
    generated by torch._dynamo.
    Writes to ./speedups_fx2trt.csv
    """
    return speedup_experiment(args, model_iter_fn, model, example_inputs)


# TODO: CompilerProfiler is deprecated, remove this
def recompile_profiler_experiment(args, model_iter_fn, model, example_inputs):
    prof = torch._dynamo.utils.CompilerProfiler()
    opt_model_iter_fn = torch._dynamo.optimize(prof, nopython=args.nopython)(
        model_iter_fn
    )
    opt_model_iter_fn(model, example_inputs)
    write_outputs(
        output_filename, ["model", "profiler report"], [current_name, prof.report()]
    )
    met = prof.get_metrics()
    guard_failures = len(met["guard_failures"])
    return [guard_failures]


def randomize_input(inputs):
    if isinstance(inputs, (list, tuple)):
        return type(inputs)([randomize_input(x) for x in inputs])
    elif isinstance(inputs, torch.Tensor):
        if inputs.dtype in (torch.float32, torch.float64):
            torch._dynamo.utils.counters["randomize_input"]["times"] += 1
            return torch.randn_like(inputs)
        elif inputs.dtype == torch.int64:
            # Note: we can not simply tune integer tensors as follows
            #   `return torch.randint_like(inputs, high=inputs.max().item())`
            # This may break some invariants between tensors.
            # E.g. in embedding lookup case, one tensor is the length
            # and another is an indices tensor.
            return inputs
        else:
            raise RuntimeError(
                f"randomize_input need support tensor of type {inputs.dtype}"
            )
    else:
        raise RuntimeError(
            f"randomize_input can not handle input of type {type(inputs)}"
        )


def maybe_mark_step(args):
    if args.trace_on_xla:
        xm.mark_step()


def latency_experiment(args, model_iter_fn, model, example_inputs, mark, **kwargs):
    """
    Measure latency on a specific backend.
    """

    timings = np.zeros((args.repeat,), np.float64)
    # if we randomize the input, we should also check the result is correct
    should_randomize_input = args.randomize_input

    import contextlib

    from torch._inductor.utils import maybe_profile

    @contextlib.contextmanager
    def maybe_mark_profile(*args, **kwargs):
        prof: torch.profiler.profile = kwargs.pop("p", None)
        mark = kwargs.pop("mark", None)
        if prof:
            with torch.profiler.record_function(mark):
                yield
        else:
            yield

    times = args.iterations_per_run

    with maybe_profile(args.export_profiler_trace, **args.profile_details) as p:
        for rep in trange(args.repeat, desc="running benchmark"):
            inputs = (
                randomize_input(copy.deepcopy(example_inputs))
                if should_randomize_input
                else example_inputs
            )
            # need call mark_step to perform the computation
            # on randomize_input. Otherwise the first call using the
            # inputs will incur high penalty then the next one.
            maybe_mark_step(args)

            with maybe_mark_profile(p=p, mark=mark):
                timings[rep], actual_output = timed(
                    model,
                    model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                )

    if args.export_profiler_trace:
        name = args.profiler_trace_name + "_" + model.name
        if hasattr(args, "rank"):
            name += f"_rank_{args.rank}"
        name += ".json"
        name = os.path.join(torch._dynamo.config.base_dir, name)
        p.export_chrome_trace(name)
    return timings


# TODO: This seems to be specifically triggered by torchao testing
def latency_experiment_summary(suite_name, args, model, timings, **kwargs):
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    if args.dump_raw_metrics:
        np.save(
            f"{output_filename[:-4]}-raw_timings-{current_name}-{current_device}.npy",
            timings,
        )

    first_headers = ["dev", "name", "batch_size"]
    first_fields = [current_device, current_name, current_batch_size]
    if "tag" in kwargs:
        first_headers.append("tag")
        first_fields.append(kwargs["tag"])
    headers = first_headers + ["speedup", "abs_latency"]
    row = first_fields + [float(speedup), median[1] * 1000]
    msg = f"{median[0] * 1000} ms, {median[1] * 1000} ms, {speedup:.3f}x"
    if args.baseline:
        headers.extend(
            [
                "baseline",
                "speedup_vs_baseline",
            ]
        )
        df = pd.read_csv(args.baseline)
        try:
            baseline_speedup = df[df["name"] == current_name]["speedup"].item()
            row.extend([baseline_speedup, speedup / baseline_speedup])
            msg = f"{baseline_speedup:.3f}x -> {speedup:.3f}x [{speedup / baseline_speedup:.3f}x]"
        except (KeyError, ZeroDivisionError):
            row.extend(
                [
                    0.0,
                    0.0,
                ]
            )
    if "compilation_latency" in kwargs:
        headers += [
            "compilation_latency",
            "compression_ratio",
            "eager_peak_mem",
            "dynamo_peak_mem",
        ]
        row.append(kwargs["compilation_latency"])
        row.append(kwargs["compression_ratio"])
        row.append(kwargs["eager_peak_mem"])
        row.append(kwargs["dynamo_peak_mem"])

    if "cache_lookup_latency" in kwargs:
        headers.append("cache_lookup_latency")
        row.append(kwargs["cache_lookup_latency"])

    if "dynamo_stats" in kwargs:
        for k, v in kwargs["dynamo_stats"].items():
            headers.append(k)
            row.append(v)
    write_outputs(
        output_filename,
        headers,
        row,
    )
    c_headers, c_data = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    assert output_filename.find(".csv") > 0, (
        f"expected output_filename to be a .csv, but got {output_filename}"
    )
    write_outputs(
        output_filename[:-4] + "_compilation_metrics.csv",
        first_headers + c_headers,
        first_fields + c_data,
    )

    # Hypothetically you can use this from other places, but it's currently
    # inaccessible, and when this assert fails you need to update the
    # event_name here to account for the other cases you are using this
    assert any([args.quantization, args.optimus])
    output_signpost(
        dict(zip(headers, row)),
        args,
        suite_name,
    )

    return msg


def speedup_experiment(args, model_iter_fn, model, example_inputs, **kwargs):
    """
    Measure speedups over eager.

    Writes to ./speedups.csv
    """
    timings = np.zeros((args.repeat, 2), np.float64)
    # if we randomize the input, we should also check the result is correct
    should_randomize_input = args.randomize_input

    import contextlib

    from torch._inductor.utils import maybe_profile

    @contextlib.contextmanager
    def maybe_mark_profile(*args, **kwargs):
        prof: torch.profiler.profile = kwargs.pop("p", None)
        mark = kwargs.pop("mark", None)
        if prof:
            with torch.profiler.record_function(mark):
                yield
        else:
            yield

    times = args.iterations_per_run

    # Use higher tolerance for XLA since XLA cause numerical instability when
    # graph size changes
    tolerance = args.xla_tolerance if args.trace_on_xla else 1e-4
    torch._dynamo.config.repro_tolerance = tolerance

    with maybe_profile(args.export_profiler_trace, **args.profile_details) as p:
        if args.export_aot_inductor:
            frozen_model_iter_fn = export_aot_inductor(
                model, example_inputs, args.inductor_compile_mode
            )
        elif args.export_nativert:
            frozen_model_iter_fn = export_nativert(model, example_inputs)
        elif args.torchscript_jit_trace:
            frozen_model_iter_fn = torchscript_jit_trace(model, example_inputs)
        elif args.aot_precompile:
            frozen_model_iter_fn = aot_precompile(model, example_inputs)
        else:
            if kwargs["hf_llm"]:
                # If it's an llm, we want to optimize model.forward, and use
                # the generate function
                model.forward = torch._dynamo.run(model)
                frozen_model_iter_fn = model_iter_fn
            else:
                frozen_model_iter_fn = torch._dynamo.run(model_iter_fn)

        for rep in trange(args.repeat, desc="running benchmark"):
            inputs = (
                randomize_input(copy.deepcopy(example_inputs))
                if should_randomize_input
                else example_inputs
            )
            # need call mark_step to perform the computation
            # on randomize_input. Otherwise the first call using the
            # inputs will incur high penalty then the next one.
            maybe_mark_step(args)

            # interleave the runs to handle frequency scaling and load changes
            with (
                maybe_mark_profile(p=p, mark="expected"),
                torch.compiler.set_stance("force_eager"),
            ):
                timings[rep, 0], expected_output = timed(
                    model,
                    model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                    batch_size=kwargs.get("batch_size"),
                )

            # call mark_step between the 2 calls to make the comparison fair.
            maybe_mark_step(args)

            with maybe_mark_profile(p=p, mark="actual"):
                timings[rep, 1], actual_output = timed(
                    model,
                    frozen_model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                )

    if args.export_profiler_trace:
        name = args.profiler_trace_name + "_" + model.name
        if hasattr(args, "rank"):
            name += f"_rank_{args.rank}"
        if args.export_perfdoctor and trace_handler:
            trace_handler(name, p)
        else:
            name += ".json"
            name = os.path.join(torch._dynamo.config.base_dir, name)
            p.export_chrome_trace(name)

    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    if args.dump_raw_metrics:
        np.save(
            f"{output_filename[:-4]}-raw_timings-{current_name}-{current_device}.npy",
            timings,
        )

    first_headers = ["dev", "name", "batch_size"]
    first_fields = [current_device, current_name, current_batch_size]
    if "tag" in kwargs:
        first_headers.append("tag")
        first_fields.append(kwargs["tag"])
    headers = first_headers + ["speedup", "abs_latency"]
    row = first_fields + [float(speedup), median[1] * 1000]
    msg = f"{speedup:.3f}x"
    if args.baseline:
        headers.extend(
            [
                "baseline",
                "speedup_vs_baseline",
            ]
        )
        df = pd.read_csv(args.baseline)
        try:
            baseline_speedup = df[df["name"] == current_name]["speedup"].item()
            row.extend([baseline_speedup, speedup / baseline_speedup])
            msg = f"{baseline_speedup:.3f}x -> {speedup:.3f}x [{speedup / baseline_speedup:.3f}x]"
        except (KeyError, ZeroDivisionError):
            row.extend(
                [
                    0.0,
                    0.0,
                ]
            )
    if "compilation_latency" in kwargs:
        headers += [
            "compilation_latency",
            "compression_ratio",
            "eager_peak_mem",
            "dynamo_peak_mem",
        ]
        row.append(kwargs["compilation_latency"])
        row.append(kwargs["compression_ratio"])
        row.append(kwargs["eager_peak_mem"])
        row.append(kwargs["dynamo_peak_mem"])

    if "cache_lookup_latency" in kwargs:
        headers.append("cache_lookup_latency")
        row.append(kwargs["cache_lookup_latency"])

    if "dynamo_stats" in kwargs:
        for k, v in kwargs["dynamo_stats"].items():
            headers.append(k)
            row.append(v)
    write_outputs(
        output_filename,
        headers,
        row,
    )
    c_headers, c_data = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    assert output_filename.find(".csv") > 0, (
        f"expected output_filename to be a .csv, but got {output_filename}"
    )
    write_outputs(
        output_filename[:-4] + "_compilation_metrics.csv",
        first_headers + c_headers,
        first_fields + c_data,
    )

    output_signpost(
        dict(zip(headers, row)),
        args,
        get_suite_from_model_iter_fn(model_iter_fn),
    )

    return msg


def overhead_experiment(*args, model_iter_fn):
    """
    Measure overheads of TorchDynamo by running with no backend (only
    eager+FX), and reporting speedup/slowdown over eager.

    Writes to ./overheads.csv
    """
    return speedup_experiment(*args, model_iter_fn)


def print_fx(gm, example_inputs):
    print(gm.graph)
    return gm


def print_aten_ops(gm, example_inputs):
    from functorch.compile import aot_module

    def trace_printer(gm, _):
        print(gm.graph)
        return gm

    return aot_module(gm, fw_compiler=trace_printer, bw_compiler=trace_printer)


def baselines(models, model_iter_fn, example_inputs, args):
    """
    Common measurement code across all baseline experiments.
    """
    models = list(models)
    for idx, (name, model) in enumerate(models):
        if idx == 0:
            result0 = model_iter_fn(model, example_inputs)
        elif model is not None:
            try:
                result = model_iter_fn(model, example_inputs)
                if same(result0, result):
                    continue
                print(name, "is INCORRECT")
            except Exception:
                log.exception("error checking %s", name)
            models[idx] = (name, None)
    timings = np.zeros((args.repeat, len(models)), np.float64)
    timings.fill(1.0e10)
    for rep in range(args.repeat):
        for idx, (name, model) in enumerate(models):
            if model is not None:
                try:
                    timings[rep, idx] = timed(model, model_iter_fn, example_inputs)
                except Exception:
                    pass
    pvalue = [
        ttest_ind(timings[:, 0], timings[:, i]).pvalue
        for i in range(1, timings.shape[1])
    ]
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1:]
    for idx, (name, model) in enumerate(models[1:]):
        if model is None:
            speedup[idx] = 0.0
    result = " ".join(
        [
            format_speedup(s, p, m is not None)
            for s, p, m in zip(speedup, pvalue, [m for n, m in models[1:]])
        ]
    )
    write_outputs(
        output_filename,
        ("dev", "name", "batch_size") + tuple(n for n, m in models[1:]),
        [current_device, current_name, current_batch_size]
        + [f"{x:.4f}" for x in speedup],
    )
    return result


def xla(args, model_iter_fn, model, example_inputs):
    xla_dev = xm.xla_device(devkind=current_device)
    model_xla = copy.deepcopy(model).to("cpu").to(device=xla_dev)
    example_inputs_xla = tree_map_only(
        torch.Tensor, lambda x: x.to("cpu").to(device=xla_dev), example_inputs
    )
    for _ in range(3):  # warmup
        timed(model, model_iter_fn, example_inputs)
        timed(model_xla, model_iter_fn, example_inputs_xla)
    timings = np.zeros((args.repeat, 2), np.float64)
    timings.fill(1.0e10)
    for rep in range(args.repeat):
        timings[rep, 0] = timed(model, model_iter_fn, example_inputs)
        timings[rep, 1] = timed(model_xla, model_iter_fn, example_inputs_xla)

    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    time_baseline, time_xla = np.median(timings, axis=0)
    speedup = time_baseline / time_xla
    write_outputs(
        output_filename,
        ("dev", "name", "batch_size", "speedup", "time_baseline", "time_xla"),
        [
            current_device,
            current_name,
            current_batch_size,
            speedup,
            time_baseline,
            time_xla,
        ],
    )
    return format_speedup(speedup, pvalue)


def try_script(model, example_inputs):
    try:
        return torch.jit.script(model)
    except Exception:
        return None


def _produce_dynamic_shapes_for_export(path, x):
    # mark_dynamic() is ignored for export.
    # use this to produce dynamic_shapes spec instead.
    from torch.export.dynamic_shapes import Dim

    if not isinstance(x, torch.Tensor):
        return None
    return dict.fromkeys(getattr(x, "_dynamo_dynamic_indices", {}), Dim.AUTO)


class AOTInductorModelCache:
    cache: dict[weakref.ref, tuple[Any, float]] = {}

    @classmethod
    def load(cls, model, example_inputs, mode):
        import torch._inductor
        from torch.export.dynamic_shapes import _combine_args, _tree_map_with_path

        key = weakref.ref(model)
        if key not in cls.cache:
            # Register the output dataclass to pytree
            example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
            with torch.no_grad():
                # copy.deepcopy is required to prevent any surprising side-effect,
                # see https://github.com/pytorch/pytorch/issues/113029
                # This will cause memory stats to be overshadowed by this eager run.
                # To fix that, memory stats will be reset later.
                example_outputs = copy.deepcopy(model)(*example_args, **example_kwargs)

            if pytree.is_namedtuple_instance(example_outputs):
                typ = type(example_outputs)
                pytree._register_namedtuple(
                    typ,
                    serialized_type_name=f"{typ.__module__}.{typ.__name__}",
                )
            else:
                _register_dataclass_output_as_pytree(example_outputs)

            combined_args = _combine_args(model, example_args, example_kwargs)
            dynamic_shapes = _tree_map_with_path(
                _produce_dynamic_shapes_for_export, combined_args
            )

            # delete example_outputs and reset memory stats here
            del example_outputs
            if current_device == "cuda":
                empty_gpu_cache(current_device)
                torch.cuda.reset_peak_memory_stats()
                pre_clone_memory_used = torch.cuda.max_memory_allocated()
            elif current_device == "hpu":
                torch.hpu.reset_peak_memory_stats()
                pre_clone_memory_used = torch.hpu.max_memory_allocated()

            # Clone the model pre-exporting.  This prevents scenarios observed in a few
            # models, where the forward pass modifies model state while exporting, and
            # FakeTensors are thus saved as model data members.  This invalidates model
            # reuse in eager mode, so it's safest to export a model clone.
            model_clone = copy.deepcopy(model)

            # Since CPU doesn't monitor max memory allocation, anything measuring peak
            # memory will miss our transient model clone on CPU anyway.
            #
            # The justification for tracking this value (in order to remove it from the
            # AOTInductor memory measurements) is that normal usage of AOTInductor would
            # not clone the model, since the eager model would be unused post-export.
            clone_memory_used = 0.0
            if current_device == "cuda":
                clone_memory_used = (
                    torch.cuda.max_memory_allocated() - pre_clone_memory_used
                ) / 1e9
            elif current_device == "hpu":
                clone_memory_used = (
                    torch.hpu.max_memory_allocated() - pre_clone_memory_used
                ) / 1e9

            inductor_configs = {}
            if mode == "max-autotune":
                inductor_configs["max_autotune"] = True
            ep = torch.export.export(
                model_clone,
                example_args,
                example_kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            with torch.no_grad():
                package_path = torch._inductor.aoti_compile_and_package(
                    ep, inductor_configs=inductor_configs
                )  # type: ignore[arg-type]

            cls.cache[key] = (
                torch._inductor.aoti_load_package(package_path),
                clone_memory_used,
            )

        return cls.cache[key][0]

    @classmethod
    def get_excess_memory(cls, model) -> float:
        return cls.cache.get(weakref.ref(model), (None, 0.0))[1]


class NativeRTCache:
    cache: dict[weakref.ref, Any] = {}

    @classmethod
    def load(cls, model, example_inputs):
        from torch.export.dynamic_shapes import _combine_args, _tree_map_with_path

        key = weakref.ref(model)
        if key not in cls.cache:
            example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
            example_outputs = model(*example_args, **example_kwargs)
            _register_dataclass_output_as_pytree(example_outputs)

            combined_args = _combine_args(model, example_args, example_kwargs)
            dynamic_shapes = _tree_map_with_path(
                _produce_dynamic_shapes_for_export, combined_args
            )

            ep = torch.export.export(
                model, example_args, example_kwargs, dynamic_shapes=dynamic_shapes
            )
            ep = ep.run_decompositions({})
            with tempfile.NamedTemporaryFile(delete=False) as f:
                torch.export.pt2_archive._package.package_pt2(
                    f, exported_programs={"forward": ep}
                )
                filename = f.name
            cls.cache[key] = PyModelRunner(filename, "forward")

        return cls.cache[key]


class JitTracedCache:
    cache: dict[weakref.ref, Any] = {}

    @classmethod
    def load(cls, model, example_inputs):
        key = weakref.ref(model)
        if key not in cls.cache:
            example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
            if example_args:
                jit_traced_module = torch.jit.trace(
                    model, example_inputs=example_args, strict=False
                )
            else:
                jit_traced_module = torch.jit.trace(
                    model, example_kwarg_inputs=example_kwargs, strict=False
                )

            cls.cache[key] = jit_traced_module

        return cls.cache[key]


def export(model, example_inputs):
    from torch.export.dynamic_shapes import _combine_args, _tree_map_with_path

    example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
    example_outputs = model(*example_args, **example_kwargs)
    _register_dataclass_output_as_pytree(example_outputs)

    combined_args = _combine_args(model, example_args, example_kwargs)
    dynamic_shapes = _tree_map_with_path(
        _produce_dynamic_shapes_for_export, combined_args
    )

    # NOTE: if args.export is ever enabled for --performance mode (rather than solely
    # --accuracy), we'll need to clone the model and subtract out extra memory usage, as
    # done in AOTInductorModelCache.
    ep = torch.export.export(
        model, example_args, example_kwargs, dynamic_shapes=dynamic_shapes, strict=True
    )

    def opt_export(_, example_inputs):
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return ep.module()(*example_args, **example_kwargs)

    return opt_export


def aot_precompile(model, example_inputs):
    example_args, example_kwargs = _normalize_bench_inputs(example_inputs)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_path = f.name

    with fresh_cache(), torch._dynamo.config.patch("enable_aot_compile", True):
        compiled_fn = torch.compile(
            model,
            fullgraph=True,
            options={"guard_filter_fn": lambda guards: [False for _ in g
```



## High-Level Overview


This Python file contains 13 class(es) and 153 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CI`, `Stats`, `AOTInductorModelCache`, `NativeRTCache`, `JitTracedCache`, `TimeOutException`, `DummyGradScaler`, `BenchmarkRunner`, `LinearModel`

**Functions defined**: `load_yaml_file`, `flatten`, `maybe_list_to_set`, `model_specified_by_path`, `load_model_from_path`, `write_outputs`, `output_csv`, `output_json`, `get_suite_from_model_iter_fn`, `output_signpost`, `nothing`, `patch_torch_manual_seed`, `deterministic_torch_manual_seed`, `empty_gpu_cache`, `synchronize`, `summarize_graph_break`, `print_summary`, `print_summary_table`, `tensor_is_on_xla`, `visit`

**Key imports**: annotations, argparse, collections, contextlib, copy, csv, dataclasses, functools, gc, importlib


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `collections`
- `contextlib`
- `copy`
- `csv`
- `dataclasses`
- `functools`
- `gc`
- `importlib`
- `itertools`
- `json`
- `logging`
- `os`
- `platform`
- `random`
- `shutil`
- `signal`
- `subprocess`
- `sys`
- `tempfile`
- `time`
- `weakref`
- `typing`: Any, NamedTuple, Optional, overload, TYPE_CHECKING, TypeVar
- `unittest.mock`: MagicMock
- `numpy as np`
- `pandas as pd`
- `psutil`
- `yaml`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs
- **Database**: May involve SQL - watch for injection vulnerabilities

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/dynamo`):

- [`timm_models_list_cpu.txt_docs.md`](./timm_models_list_cpu.txt_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test.py_docs.md`](./test.py_docs.md)
- [`benchmarks.py_docs.md`](./benchmarks.py_docs.md)
- [`check_graph_breaks.py_docs.md`](./check_graph_breaks.py_docs.md)
- [`check_csv.py_docs.md`](./check_csv.py_docs.md)
- [`all_torchbench_models_list.txt_docs.md`](./all_torchbench_models_list.txt_docs.md)
- [`check_accuracy.py_docs.md`](./check_accuracy.py_docs.md)
- [`torchbench_models_list_cpu.txt_docs.md`](./torchbench_models_list_cpu.txt_docs.md)
- [`timm_models.py_docs.md`](./timm_models.py_docs.md)


## Cross-References

- **File Documentation**: `common.py_docs.md`
- **Keyword Index**: `common.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
