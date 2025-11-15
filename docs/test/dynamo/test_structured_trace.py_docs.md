# Documentation: `test/dynamo/test_structured_trace.py`

## File Metadata

- **Path**: `test/dynamo/test_structured_trace.py`
- **Size**: 102,391 bytes (99.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import copy
import functools
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import unittest.mock
from contextlib import contextmanager

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging.structured
import torch.distributed as dist
import torch.fx as fx
from torch._inductor.test_case import TestCase
from torch._logging._internal import TorchLogsFormatter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import find_free_port, xfailIfS390X
from torch.testing._internal.triton_utils import requires_cuda_and_triton


if torch.distributed.is_available():
    from torch.testing._internal.distributed.fake_pg import FakeStore

HAS_TLPARSE = shutil.which("tlparse") is not None
requires_tlparse = unittest.skipUnless(HAS_TLPARSE, "requires tlparse")
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


def example_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(1000, 1000))
    return output


def example_training_fn(a):
    output = a.mul(torch.ones(1000, 1000, requires_grad=True))
    output = output.add(torch.ones(1000, 1000))
    output.sum().backward()
    return output


def dynamo_error_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(10, 10))
    return output


def inductor_error_fn(a):
    output = torch.round(a)
    return output


def inductor_schedule_fn(a):
    output = a.add(torch.ones(1000, 1000, device="cuda"))
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)


def replace_dynamic(buffer, key):
    return re.sub(r'("' + key + r'":\s*)(\d+\.\d+)', r"\1<dynamic>", buffer)


class StructuredTraceTestingFilter(logging.Filter):
    def __init__(self, match_name=None):
        self.match_name = match_name

    def filter(self, record):
        if "str" in record.metadata:
            return False
        if self.match_name is not None:
            if "artifact" in record.metadata:
                if self.match_name != record.metadata["artifact"]["name"]:
                    return False
            elif self.match_name not in record.metadata:
                return False
        return True


class ChromiumEventFilter(logging.Filter):
    def filter(self, record):
        return "chromium_event" not in record.metadata


class StructuredTracePayloadFormatter(logging.Formatter):
    def format(self, record):
        return record.payload.strip()


class StructuredTraceTestingFormatter(logging.Formatter):
    def format(self, record):
        metadata = copy.deepcopy(record.metadata)

        # Stub out values that are not stable across runs
        # TODO: Check that these match schema
        if "has_payload" in metadata:
            metadata["has_payload"] = "HASH"
        if "dynamo_start" in metadata:
            metadata["dynamo_start"]["stack"] = "STACK"
        if "inductor_output_code" in metadata:
            metadata["inductor_output_code"]["filename"] = "FILENAME"
            if "file_path" in metadata["inductor_output_code"]:
                metadata["inductor_output_code"]["file_path"] = "FILENAME"
        if "stack" in metadata:
            metadata["stack"] = "STACK"
        if "compilation_metrics" in metadata:
            metadata["compilation_metrics"] = "METRICS"
        if "bwd_compilation_metrics" in metadata:
            metadata["bwd_compilation_metrics"] = "METRICS"
        if "compilation_metrics_runtime" in metadata:
            metadata["compilation_metrics_runtime"] = "METRICS"
        if "bwd_compilation_metrics_runtime" in metadata:
            metadata["bwd_compilation_metrics_runtime"] = "METRICS"
        if "describe_storage" in metadata:
            metadata["describe_storage"]["describer_id"] = "ID"
        if "describe_tensor" in metadata:
            metadata["describe_tensor"]["describer_id"] = "ID"
            if "view_func" in metadata["describe_tensor"]:
                metadata["describe_tensor"]["view_func"] = "VIEW_FUNC"
        if "describe_source" in metadata:
            metadata["describe_source"]["describer_id"] = "ID"
        if (
            (k := "create_symbol") in metadata
            or (k := "guard_added_fast") in metadata
            or (k := "create_unbacked_symbol") in metadata
        ):
            metadata[k]["user_stack"] = "STACK"
            metadata[k]["stack"] = "STACK"

        if "dump_file" in metadata:
            # Don't include the actually key number, that's sensitive to other
            # test runs
            metadata["dump_file"]["name"] = "<eval_with_key>"
            return (
                json.dumps(metadata)
                + "\n"
                + "\n".join(l.rstrip() for l in record.payload.splitlines())
            )

        return json.dumps(metadata)


trace_log = logging.getLogger("torch.__trace")

chrome_event_filter = ChromiumEventFilter()


def show_chrome_events(fn):
    """
    Don't hide chrome events for this test
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.handler.removeFilter(chrome_event_filter)
        return fn(self, *args, **kwargs)

    return wrapper


class StructuredTraceTest(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._logging.structured.INTERN_TABLE.clear()
        self.buffer = io.StringIO()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTraceTestingFormatter())
        self.handler.addFilter(StructuredTraceTestingFilter())
        self.handler.addFilter(chrome_event_filter)
        trace_log.addHandler(self.handler)

        self.raw_file = tempfile.NamedTemporaryFile(
            mode="w", delete=True
        )  # set this to False to keep temporary files
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        trace_log.addHandler(self.raw_handler)

    def tearDown(self):
        trace_log.removeHandler(self.handler)
        trace_log.removeHandler(self.raw_handler)
        self.raw_file.close()
        trace_log.setLevel(self.old_level)

    def assertParses(self):
        out = tempfile.mkdtemp()
        try:
            subprocess.check_call(
                [
                    "tlparse",
                    "-o",
                    out,
                    "--overwrite",
                    "--no-browser",
                    "--strict",
                    self.raw_file.name,
                ]
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_compile_id_serialization_deserialization(self):
        cid = torch._guards.CompileId(
            frame_id=1,
            frame_compile_id=2,
        )
        assert cid == torch._guards.CompileId.from_string(str(cid))

        cid = torch._guards.CompileId(
            compiled_autograd_id=1,
            frame_id=2,
            frame_compile_id=3,
        )
        assert cid == torch._guards.CompileId.from_string(str(cid))

        cid = torch._guards.CompileId(
            compiled_autograd_id=1,
            frame_id=None,
            frame_compile_id=None,
        )
        assert cid == torch._guards.CompileId.from_string(str(cid))

        for bad_cid in ["-/-", "-/1", "1/-", "!1/2", "!1/-/-"]:
            with self.assertRaises(ValueError):
                torch._guards.CompileId.from_string(bad_cid)

    @requires_cuda_and_triton
    def test_schedule(self):
        fn_opt = torch.compile(inductor_schedule_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "triton_kernel_info", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"compilation_metrics_runtime": "METRICS", "frame_id": 0, "frame_compile_id": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_cuda_and_triton
    def test_cudagraphs(self):
        fn_opt = torch.compile(mode="reduce-overhead")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "triton_kernel_info", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"compilation_metrics_runtime": "METRICS", "frame_id": 0, "frame_compile_id": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_recompiles(self):
        def fn(x, y):
            return torch.add(x, y)

        fn_opt = torch.compile(fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000), torch.ones(1000, 1000))
        fn_opt(torch.ones(1000, 1000), 1)

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['y']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1000, 1000], "l_y_": [1000, 1000], "add": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "recompile_reasons", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s48", "val": "1", "vr": "[-int_oo, int_oo]", "source": "L['y']", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1000, 1000], "add": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_example_fn(self):
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000))
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000], "ones_1": [1000, 1000], "output_1": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_joint_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_example_training_fn(self):
        fn_opt = torch.compile(example_training_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, requires_grad=True))
        buffer = self.buffer.getvalue()
        buffer = replace_dynamic(buffer, "inductor_compile_time_s")
        buffer = replace_dynamic(buffer, "code_gen_time_s")
        buffer = replace_dynamic(buffer, "structured_logging_overhead_s")
        self.assertExpectedInline(
            buffer,
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['___stack1']"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"dynamo_cpp_guards_str": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['___stack0']"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['___stack0']"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_output_graph": {"sizes": {"l_stack0_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000], "sum_1": []}}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"aot_joint_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"aot_forward_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"aot_backward_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"compilation_metrics": "METRICS", "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"bwd_compilation_metrics": "METRICS", "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['output']"}, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"compilation_metrics": "METRICS", "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_dynamo_error(self):
        try:
            fn_opt = torch.compile(dynamo_error_fn, backend="inductor")
            fn_opt(*ARGS)
        except Exception:
            pass
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_error", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_inductor_error(self):
        import torch._inductor.lowering

        def throw(x):
            raise AssertionError

        # inject an error in the lowerings
        dict_entries = {}
        for x in list(torch._inductor.lowering.lowerings.keys()):
            if "round" in x.__name__:
                dict_entries[x] = throw

        with unittest.mock.patch.dict(torch._inductor.lowering.lowerings, dict_entries):
            try:
                fn_opt = torch.compile(inductor_error_fn, backend="inductor")
                fn_opt(*ARGS)
            except Exception:
                pass

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "output": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_joint_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "dynamo_error", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_distributed()
    @requires_cuda_and_triton
    def test_ddp_graphs(self):
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(1024, 1024),
                    torch.nn.Linear(1024, 1024),
                )

            def forward(self, x):
                return self.layers(x)

        # TODO: this isn't safely bracketed, will leak
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)

        model = DDP(ToyModel().to("cuda:0"), device_ids=[0], bucket_cap_mb=4)
        ddp_model = torch.compile(model, backend="inductor")

        ddp_model(torch.randn(1024, 1024, device="cuda:0"))

        dist.destroy_process_group()

        if not torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertExpectedInline(
                self.buffer.getvalue(),
                """\
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1024, 1024], "l__self___layers_0": [1024, 1024], "l__self___layers_1": [1024, 1024]}}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_graph": {}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_child": {"name": "submod_0"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_child": {"name": "submod_1"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_joint_graph": {}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_bypass", "encoding": "json"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_pre_grad_graph", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "after_pre_grad_graph", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_joint_graph": {}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "torch._functorch.config", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "before_post_grad_graph", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "inductor_post_grad_graph", "encoding": "string"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME", "file_path": "FILENAME"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_bypass", "encoding": "json"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
            )
        else:
            self.assertExpectedInline(
                self.buffer.getvalue(),
                """\
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "dynamo_hint_overrides": {}, "is_leaf": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['args'][0]"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['self']._modules['layers']._modules['0']._parameters['weight']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 4096}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024], "dynamo_hint_overrides": {}, "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['self']._modules['layers']._modules['0']._parameters['bias']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 2, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id
```



## High-Level Overview


This Python file contains 12 class(es) and 82 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StructuredTraceTestingFilter`, `ChromiumEventFilter`, `StructuredTracePayloadFormatter`, `StructuredTraceTestingFormatter`, `StructuredTraceTest`, `ToyModel`, `MySin`, `Step`, `CollectiveModule`, `SimpleModule`, `MixedModule`, `Mixed`

**Functions defined**: `example_fn`, `example_training_fn`, `dynamo_error_fn`, `inductor_error_fn`, `inductor_schedule_fn`, `replace_dynamic`, `__init__`, `filter`, `filter`, `format`, `format`, `show_chrome_events`, `wrapper`, `setUp`, `tearDown`, `assertParses`, `test_compile_id_serialization_deserialization`, `test_schedule`, `test_cudagraphs`, `test_recompiles`

**Key imports**: copy, functools, io, json, logging, os, re, shutil, subprocess, tempfile


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `functools`
- `io`
- `json`
- `logging`
- `os`
- `re`
- `shutil`
- `subprocess`
- `tempfile`
- `unittest.mock`
- `contextlib`: contextmanager
- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`
- `torch._logging.structured`
- `torch.distributed as dist`
- `torch.fx as fx`
- `torch._inductor.test_case`: TestCase
- `torch._logging._internal`: TorchLogsFormatter
- `torch.nn.parallel`: DistributedDataParallel as DDP
- `torch.testing._internal.common_utils`: find_free_port, xfailIfS390X
- `torch.testing._internal.triton_utils`: requires_cuda_and_triton
- `torch.testing._internal.distributed.fake_pg`: FakeStore
- `torch._inductor.lowering`
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch._inductor.debug`: log_collective_schedule


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/test_structured_trace.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_structured_trace.py_docs.md`
- **Keyword Index**: `test_structured_trace.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
