# Documentation: `test/distributed/test_dynamo_distributed.py`

## File Metadata

- **Path**: `test/distributed/test_dynamo_distributed.py`
- **Size**: 85,702 bytes (83.69 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import contextlib
import copy
import functools
import logging
import random
import unittest
from contextlib import contextmanager
from datetime import timedelta
from io import StringIO
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case
import torch.distributed as dist
import torch.optim as optim
from torch import nn
from torch._C import FileCheck
from torch._dynamo import config
from torch._dynamo.backends.distributed import DDPOptimizer
from torch._dynamo.comptime import comptime
from torch._dynamo.testing import collect_results
from torch._dynamo.utils import same
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch.compiler import set_enable_guard_collectives
from torch.distributed._functional_collectives import _maybe_wrap_tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    DynamoDistributedMultiProcTestCase,
    DynamoDistributedSingleProcTestCase,
    import_transformers_or_skip,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.testing._internal.triton_utils import requires_cuda_and_triton


log = logging.getLogger(__name__)


def reset_rng_state():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


@contextmanager
def enable_guard_collectives():
    old = set_enable_guard_collectives(True)
    try:
        yield
    finally:
        set_enable_guard_collectives(old)


class ToyModel(nn.Module):
    def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None):
        super().__init__()
        self.ctx_manager = ctx_manager
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
        )

    def forward(self, inputs):
        if self.ctx_manager is not None:
            with self.ctx_manager():
                return self.net(inputs)
        else:
            return self.net(inputs)


def get_model(
    device, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
):
    m = ToyModel(
        in_feat=in_feat,
        hidden_feat=hidden_feat,
        out_feat=out_feat,
        ctx_manager=ctx_manager,
    ).to(device)
    m.apply(init_weights)
    inputs = torch.rand(bsz, in_feat).to(device)
    outputs = m(inputs)
    return m, inputs, outputs


class MutatingModel(nn.Module):
    def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None):
        super().__init__()
        self.ctx_manager = ctx_manager
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
        )
        self.state = 1

    def forward(self, inputs):
        self.state = 2
        return self.net(inputs) * self.state


def get_mutating_model(
    device, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
):
    m = MutatingModel(
        in_feat=in_feat,
        hidden_feat=hidden_feat,
        out_feat=out_feat,
        ctx_manager=ctx_manager,
    ).to(device)
    m.apply(init_weights)
    inputs = torch.rand(bsz, in_feat).to(device)
    outputs = m(inputs)
    return m, inputs, outputs


class ForcedGetAttrMod(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.__dict__["forced_linear"] = torch.nn.Linear(1, 1).to(device=device)
        self.counter = 0

    def forward(self, x):
        self.counter += 1
        return x * self.linear(x) * self.forced_linear.weight


def get_forced_getattr_module(device):
    mod = ForcedGetAttrMod(device).to(device=device)
    x = torch.randn(1, 1, device=device)
    return mod, x, mod(x)


class ToyInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [nn.Linear(100, 100), nn.Linear(100, 100)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.layers(inputs)


class ToyOuterModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.layers = [ToyInnerModel().to(device) for _ in range(2)]
        self.layers = nn.Sequential(
            self.layers[0], nn.ReLU(), self.layers[1], nn.ReLU()
        )

    def forward(self, inputs):
        return self.layers(inputs)


def get_toy_model_for_activation_checkpointing(device):
    m = ToyOuterModel(device).to(device)
    m.apply(init_weights)
    inputs = torch.rand(100, 100).to(device)
    return m, inputs


def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def apply_fsdp_with_checkpointing(
    model, wrap_policy, checkpoint_policy, use_activation_checkpointing=True
):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )

    model = FSDP(
        copy.deepcopy(model), auto_wrap_policy=wrap_policy, use_orig_params=True
    )
    if use_activation_checkpointing:
        checkpoint_wrapper_fn = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper_fn,
            check_fn=checkpoint_policy,
        )
    return model


def get_custom_model(device):
    class MyCustomLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(512, 512))

        def forward(self, x):
            tmp = torch.mm(x, self.weight.t())
            # test an edge case where torch.where.scalar was decomposed to aten.where.self(tensor, tensor, tensor)
            # and the tensors T(0.4) and T(0.5) were not wrapped in FakeTensors during DDPOptimizer compilation
            return tmp + torch.where(tmp < 0.5, 0.3, 0.6)

    class MyLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(512, 512)

        def forward(self, x):
            return self.linear(x)

    class MyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mods = [
                (MyLinear(), torch.nn.ReLU()),
                # sandwich the custom in the middle so it comes before and after
                (MyCustomLinear(), torch.nn.ReLU()),
                (MyLinear(), torch.nn.ReLU()),
            ]
            self.seq = torch.nn.Sequential(*[x for items in mods for x in items])

        def forward(self, x, y):
            # test special case where the 0th bucket (layers close to graph input) is at capacity, which would
            # trigger a new bucket, but there are only trivial ops without parameters to put into the new bucket.
            # optimize this case by fusing that 'empty bucket' back together with the previous full one
            return self.seq(x + y)

    m = MyModule().to(device)
    m.apply(init_weights)
    inputs = torch.rand((512, 512)).to(device)
    # test duplicated inputs
    inputs = (inputs, inputs)
    correct_outputs = m(*inputs)
    return m, inputs, correct_outputs


def get_hf_bert(rank):
    # Note: use @import_transformers_or_skip on your test case if you use this
    # in a multiprocessing test
    try:
        from transformers import AutoModelForMaskedLM, BertConfig
    except ImportError as e:
        raise unittest.SkipTest("Unable to import transformers") from e

    device_type = (
        acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
    )
    batch_size, max_length, config, device = (
        4,
        512,
        BertConfig(),
        f"{device_type}:{rank}",
    )
    model = AutoModelForMaskedLM.from_config(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(device)
    decoder_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(
        device
    )
    inputs = {"input_ids": input_ids, "labels": decoder_ids}
    model.train()
    return model, inputs


class CheckSplitsCompiler:
    def __init__(self) -> None:
        self.compiler_called = 0

    def compile_fn(self, gm, example_inputs):
        self.compiler_called += 1
        return gm


# This simulates DDP, but it doesn't actually do any process communication;
# it just has enough properties so that the dynamo distributed optimization is
# able to optimize.  Feel free to simulate more properties as necessary.  The
# other important thing is patching _active_ddp_module, which is what actually
# triggers DDP optimization
class FakeDDP(nn.Module):
    def __init__(self, module, bucket_cap_mb=25):
        super().__init__()
        self.module = module
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

    @contextmanager
    def _inside_ddp_forward(self):
        DDP._active_ddp_module = self
        try:
            yield
        finally:
            DDP._active_ddp_module = None

    def forward(self, *inputs, **kwargs):
        if not DDP._active_ddp_module:
            with self._inside_ddp_forward():
                return self.module.forward(*inputs, **kwargs)
        else:
            return self.module.forward(*inputs, **kwargs)


def run_hf_bert_ddp(self, model, inputs, backend):
    reset_rng_state()
    correct_outputs = model(**inputs)
    correct_loss = correct_outputs.loss
    correct_loss.backward()

    reset_rng_state()
    opt_model = torch.compile(model, backend=backend)
    opt_outputs = opt_model(**inputs)
    opt_loss = opt_outputs.loss
    opt_loss.backward()

    inputs_flat = [inputs[k] for k in inputs]
    correct_results = collect_results(
        model, correct_outputs.logits, correct_loss, inputs_flat
    )
    opt_results = collect_results(opt_model, opt_outputs.logits, opt_loss, inputs_flat)
    self.assertTrue(same(correct_results, opt_results))


class TestFakeDistributedSingleProc(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    @unittest.skipIf(
        torch._inductor.config.triton.native_matmul,
        "FIXME : native matmul fails. RuntimeError: Cannot access data pointer of Tensor",
    )
    def test_hf_bert_ddp_inductor(self):
        model, inputs = get_hf_bert(0)
        model = FakeDDP(model)
        run_hf_bert_ddp(self, model, inputs, "inductor")

    @patch.object(config, "optimize_ddp", True)
    def test_hf_bert_ddp_aot_eager(self):
        model, inputs = get_hf_bert(0)
        model = FakeDDP(model)
        run_hf_bert_ddp(self, model, inputs, "aot_eager")

    @patch.object(config, "optimize_ddp", True)
    def test_issue90375(self):
        class Model(nn.Module):
            def forward(self):
                return torch.randn(3) * torch.randn(3)

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(model, backend="aot_eager")
        opt_model()

    @patch.object(config, "optimize_ddp", True)
    def test_symbol_splitting(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x):
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = x + y @ self.weight2
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512))

    @patch.object(config, "optimize_ddp", True)
    def test_ddp_optimizer_inductor_strides_dont_specialize(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc_0 = nn.Linear(768, 768)
                self.fc_1 = nn.Linear(768, 768)

            def forward(self, x):
                x = self.fc_0(x)
                x = self.fc_1(x)
                return x

        model = Model()
        model = FakeDDP(model)

        inp = torch.randn((16, 18, 768))
        inp2 = torch.randn((16, 20, 768))

        torch._dynamo.mark_dynamic(inp, 1)
        torch._dynamo.mark_dynamic(inp2, 1)

        torch._dynamo.utils.clear_compilation_metrics()
        torch._dynamo.reset()
        try:
            DDP._active_ddp_module = model
            opt_model = torch.compile(model)
            self.assertEqual(0, len(torch._dynamo.utils.get_compilation_metrics()))
            opt_model(inp)
            compile_count_before = len(torch._dynamo.utils.get_compilation_metrics())
            opt_model(inp2)
            compile_count_after = len(torch._dynamo.utils.get_compilation_metrics())
            # no recompiles
            self.assertEqual(compile_count_before, compile_count_after)
        finally:
            DDP._active_ddp_module = None

    @config.patch(optimize_ddp=True, capture_scalar_outputs=True)
    def test_unbacked_symbol_splitting_direct(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                u0, _ = y.tolist()
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = (x + y @ self.weight2) * u0
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([12, 13]))

    @config.patch(optimize_ddp=True, capture_scalar_outputs=True)
    def test_unbacked_symbol_splitting_indirect(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                u0, _ = y.tolist()
                a = torch.ones(u0)
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = (x + y @ self.weight2) * a.sum()
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([12, 13]))

    @config.patch(optimize_ddp=True, capture_scalar_outputs=True)
    def test_unbacked_symbol_splitting_torture_multi(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))
                self.weight3 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                # partition one (contains the u0 def)
                u0, _ = y.tolist()
                x = torch.cat([x, x])
                y1 = x @ self.weight1
                # partition two (contains the variable)
                y2 = y1 @ self.weight2
                a = torch.ones(u0)
                # partition three
                z = (x + y2 @ self.weight3) * a.sum()
                return z

        model = Model()
        model = FakeDDP(model, bucket_cap_mb=1)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([12, 13]))

    @config.patch(optimize_ddp=True, capture_dynamic_output_shape_ops=True)
    def test_unbacked_symbol_splitting_no_binding(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                nz = y.nonzero()
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = (x + y @ self.weight2) * (nz + 1).sum()
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([0.0, 12.0, 0.0, 11.0]))

    @patch.object(config, "optimize_ddp", True)
    def test_call_method_forward(self):
        class Model(nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                layers = []
                for _ in range(2):
                    layer = nn.ModuleList(
                        [
                            nn.LayerNorm(96),
                            nn.MultiheadAttention(
                                embed_dim=96, num_heads=4, batch_first=True
                            ),
                        ]
                    )
                    layers.append(layer)
                self.layers = nn.ModuleList(layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [Batch, Freq, Time, Feature]
                B, F, T, H = x.shape
                for m in self.layers:
                    x = x.reshape(B * F, T, H)
                    x = m[0](x)
                    x, _ = m[1].forward(x, x, x)
                    x = x.reshape(B, F, T, H)
                return x

        model = Model()
        model = FakeDDP(model)
        opt_model = torch.compile(model)
        opt_model(torch.randn(2, 129, 100, 96))


# Are these tests failing?  Check and see if TestFakeDistributedSingleProc has a
# single process version; if it's just a problem in the Dynamo distributed
# # optimizer, you should be able to repro it single process!
@requires_accelerator_dist_backend(["nccl", "xccl"])
class TestMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Note: MultiProcTestCase spawns processes per test and is slow.
    Prefer MultiThreadedTestCase for most tests. Perhaps use this one
    sparingly for integration tests.
    """

    device_type = (
        acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
    )

    @skip_if_lt_x_gpu(2)
    @config.patch(optimize_ddp=False, enable_compiler_collectives=True)
    def test_ddp_baseline_aot_eager_multiprocess(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            self.assertFalse(config.optimize_ddp)
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            m = DDP(m, device_ids=[self.rank])
            m = torch.compile(m, backend="aot_eager")
            outputs = m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    def _test_hf_bert_ddp_inductor(self, static_graph):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_hf_bert(self.rank)
            model = DDP(model, static_graph=static_graph)
            run_hf_bert_ddp(self, model, inputs, "inductor")

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_ddp_inductor(self):
        self._test_hf_bert_ddp_inductor(static_graph=False)

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_ddp_inductor_static_graph(self):
        self._test_hf_bert_ddp_inductor(static_graph=True)

    def _test_hf_bert_aot_eager(self, static_graph):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_hf_bert(self.rank)
            model = DDP(model, static_graph=static_graph)
            run_hf_bert_ddp(self, model, inputs, "aot_eager")

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    def test_hf_bert_ddp_aot_eager(self):
        self._test_hf_bert_aot_eager(static_graph=False)

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    def test_hf_bert_ddp_aot_eager_static_graph(self):
        self._test_hf_bert_aot_eager(static_graph=True)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(optimize_ddp=False, enable_compiler_collectives=True)
    def test_ddp_activation_checkpointing(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 32)
                self.fc2 = torch.nn.Linear(32, 16)
                self.fc3 = torch.nn.Linear(16, 8)

            def forward(self, inp):
                return self.fc3(self.fc2(self.fc1(inp)))

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            self.assertFalse(config.optimize_ddp)
            model = MyModel().to(device=self.device_type)

            # Activation checkpointing for Linear layers.
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            check_fn = lambda submodule: isinstance(  # noqa: E731
                submodule, torch.nn.Linear
            )
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )

            model = DDP(model)
            x = torch.randn(10, 64).to(self.device_type)
            correct_outputs = model(x)

            opt_model = torch.compile(model)
            outputs = opt_model(x)
            self.assertTrue(same(correct_outputs, outputs))

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    def test_fsdp_aot_eager(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch.compile(fsdp_m, backend="aot_eager")
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

            # Test with recursive wrapping, nested FSDP around each Linear
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            fsdp_m = FSDP(
                m,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear,)
                ),
                use_orig_params=True,
            )
            fsdp_m = torch.compile(fsdp_m, backend="aot_eager")
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @requires_cuda_and_triton
    def test_ddp_optimizer_cudagraph(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # need a large channel to trigger ddp optimizer split module
                self.CHANNELS = 640
                self.convi = nn.Conv2d(46, self.CHANNELS, 3, padding=1, bias=False)
                self.convp = nn.Conv2d(
                    self.CHANNELS, self.CHANNELS, 1, padding=0, bias=False
                )
                self.bni = nn.BatchNorm2d(self.CHANNELS)

            def forward(self, bitmap_channels):
                x = self.convi(bitmap_channels)
                x = self.bni(x)
                x = self.convp(x)
                return x

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            net = Net().to(self.rank)
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=5e-2,
            )

            net = DDP(net, device_ids=[self.rank])
            opt_net = torch.compile(net, mode="reduce-overhead")
            opt_net.train()

            for _ in range(10):
                optimizer.zero_grad()
                data = torch.randn((16, 46, 8, 8), dtype=torch.float32, device="cuda")
                opt_net(data).sum().backward()

            # 2 fwd and 2 bwd graph such that 4 graphs in total
            graph_id = (
                torch._inductor.cudagraph_trees.get_container(self.rank)
                .tree_manager.new_graph_id()
                .id
            )
            self.assertTrue(graph_id == 4)

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    def test_fsdp_setattr(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            from torch._dynamo.utils import counters

            counters.clear()
            m, inputs, correct_outputs = get_mutating_model(
                f"{self.device_type}:{self.rank}"
            )
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch.compile(fsdp_m, backend="eager", fullgraph=False)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))
            self.assertEqual(len(counters["graph_break"]), 1)
            first_graph_break = list(counters["graph_break"].keys())[0]  # noqa: RUF015
            self.assertIn("setattr() on Tensor.requires_grad", first_graph_break)

    @config.patch(inline_inbuilt_nn_modules=False)
    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    def test_fsdp_unspecialized_forced_getattr_no_inline(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            from torch._dynamo.utils import counters

            counters.clear()
            m, inputs, correct_outputs = get_forced_getattr_module(
                f"{self.device_type}:{self.rank}"
            )
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch.compile(fsdp_m, backend="eager", fullgraph=False)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    def test_fsdp_unspecialized_forced_getattr_inline(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            from torch._dynamo.utils import counters

            counters.clear()
            m, inputs, correct_outputs = get_forced_getattr_module(
                f"{self.device_type}:{self.rank}"
            )
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch.compile(fsdp_m, backend="eager", fullgraph=False)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_inductor(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch.compile(fsdp_m, backend="inductor")
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

            # Test with recursive wrapping, nested FSDP around each Linear
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            fsdp_m = FSDP(
                m,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear,)
                ),
                use_orig_params=True,
            )
            fsdp_m = torch.compile(fsdp_m, backend="inductor")
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_activation_checkpointing(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_toy_model_for_activation_checkpointing(
                f"{self.device_type}:{self.rank}"
            )
            is_inner = lambda module: isinstance(module, ToyInnerModel)  # noqa: E731
            wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=is_inner)
            model = apply_fsdp_with_checkpointing(model, wrap_policy, is_inner)
            correct_outputs = model(inputs)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
            opt_model = torch.compile(model, backend=cnt)
            outputs = opt_model(inputs)
            self.assertTrue(same(correct_outputs, outputs))
            # Each FSDP module is a separate graph
            self.assertEqual(cnt.frame_count, 2)
            self.assertTrue(
                find_first_node(cnt.graphs[0], tag_activation_checkpoint) is not None
            )

    @import_transformers_or_skip()
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    # TODO(whc) Investigate why cudagraphs breaks inductor+fsdp for hf_bert
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    @patch.object(torch._inductor.config, "fallback_random", True)
    @config.patch(enable_compiler_collectives=True)
    @unittest.skipIf(
        PLATFORM_SUPPORTS_FLASH_ATTENTION or PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Inaccurate results with fused SDPA kernels",
    )
    def test_hf_bert_fsdp(self):
        def apply_fsdp(model, wrap_policy):
            model = FSDP(
                copy.deepcopy(model), auto_wrap_policy=wrap_policy, use_orig_params=True
            )
            return model

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            for wrap_policy, test_instance in (
                (None, "FSDP without recursive wrapping"),
            ):
                print(f"Running hf_bert test for {test_instance}")
                model, inputs = get_hf_bert(self.rank)
                reset_rng_state()
                eager_model = apply_fsdp(model, wrap_policy)
                correct_outputs = eager_model(**inputs)
                correct_loss = correct_outputs.loss
                correct_loss.backward()

                reset_rng_state()
                opt_model = apply_fsdp(model, wrap_policy)
                opt_model = torch.compile(opt_model, backend="inductor")
                opt_outputs = opt_model(**inputs)
                opt_loss = opt_outputs.loss
                opt_loss.backward()

                inputs_flat = [inputs[k] for k in inputs]
                correct_results = collect_results(
                    eager_model, correct_outputs.logits, correct_loss, inputs_flat
                )
                opt_results = collect_results(
                    opt_model, opt_outputs.logits, opt_loss, inputs_flat
                )
                self.assertTrue(same(correct_results, opt_results))

    @import_transformers_or_skip()
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    # TODO(whc) Investigate why cudagraphs breaks inductor+fsdp for hf_bert
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    @patch.object(torch._inductor.config, "fallback_random", True)
    @config.patch(guard_nn_modules=True, enable_compiler_collectives=True)
    def test_hf_bert_fsdp_activation_checkpointing(self):
        from transformers.models.bert.modeling_bert import BertLayer

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            for wrap_policy, test_instance in (
                (
                    functools.partial(
                        transformer_auto_wrap_policy, transformer_layer_cls=(BertLayer,)
                    ),
                    "FSDP with recursive wrapping BertLayer instances",
                ),
            ):
                print(
                    f"Running hf_bert_activation_checkpointing test for {test_instance}"
                )
                model, inputs = get_hf_bert(self.rank)
                check_fn = lambda submodule: isinstance(  # noqa: E731
                    submodule, BertLayer
                )
                reset_rng_state()
                eager_model = apply_fsdp_with_checkpointing(
                    model, wrap_policy, check_fn
                )
                correct_outputs = eager_model(**inputs)
                correct_loss = correct_outputs.loss
                correct_loss.backward()

                reset_rng_state()
                opt_model = apply_fsdp_with_checkpointing(model, wrap_policy, check_fn)
                opt_model = torch.compile(opt_model, backend="inductor")
                opt_outputs = opt_model(**inputs)
                opt_loss = opt_outputs.loss
                opt_loss.backward()

                inputs_flat = [inputs[k] for k in inputs]
                correct_results = collect_results(
                    eager_model, correct_outputs.logits, correct_loss, inputs_flat
                )
                opt_results = collect_results(
                    opt_model, opt_outputs.logits, opt_loss, inputs_flat
                )
                self.assertTrue(same(correct_results, opt_results))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_automatic_dynamic_tensor(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):

            class SimpleModel(nn.Module):
                def __init__(self, input_size, output_size):
                    super().__init__()
                    self.linear = nn.Linear(input_size, output_size)

                def forward(self, x):
                    return self.linear(x)

            torch._dynamo.utils.clear_compilation_metrics()

            model = SimpleModel(10, 2).to(self.rank)
            model.forward = torch.compile(model.forward)
            ddp_model = DDP(model, device_ids=[self.rank])

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

            def B(s):
                return [torch.randn(s, 10), torch.randint(0, 2, (s,))]

            if self.rank == 0:
                dataloader = [B(5), B(8), B(6)]
            else:
                dataloader = [B(6), B(6), B(3)]

            for data, labels in dataloader:
                data, labels = data.to(self.rank), labels.to(self.rank)
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_automatic_dynamic_scalar(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            # TODO: This should be possible to do inside the function, but
            device = f"{self.device_type}:{self.rank}"

            @torch.compile()
            def f(x, y):
                return x + torch.ones(y, device=device).sum()

            if self.rank == 0:
                dataloader = [3, 3, 7]
            else:
                dataloader = [3, 4, 9]

            for data in dataloader:
                f(torch.randn(5, device=self.rank), data)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_automatic_dynamic_speculation_divergence(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x, y):
                zx = x.shape  # noqa: F841
                zy = y.shape  # noqa: F841
                return x.sum() + y.sum()

            if self.rank == 0:
                dataloader = [4, 4]
            else:
                dataloader = [3, 4]

            for data in dataloader:
                f(
                    torch.randn(data, device=self.rank),
                    torch.randn(data, device=self.rank),
                )

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_graph_break_empty_graph_still_collective(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x, y):
                z = y  # noqa: F841
                print("woof")
                zx = x.shape  # noqa: F841
                zy = y.shape  # noqa: F841
                return x.sum() + y.sum()

            if self.rank == 0:
                dataloader = [5, 5, 6]
            else:
                dataloader = [3, 4, 5]

            for data in dataloader:
                f(
                    torch.randn(data, device=self.rank),
                    torch.randn(data, device=self.rank),
                )

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_dim_mismatch(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x, y):
                zx = x.shape  # noqa: F841
                zy = y.shape  # noqa: F841
                return x.sum() + y.sum()

            if self.rank == 0:
                dataloader = [[4, 2]]
            else:
                dataloader = [[3]]

            for data in dataloader:
                f(
                    torch.randn(data, device=self.rank),
                    torch.randn(data, device=self.rank),
                )

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_missing_source(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(rank, xs):
                return xs[rank].sum()

            xs = []
            for _ in range(self.world_size):
                xs.append(torch.randn(10, device=self.rank))

            f(self.rank, xs)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_scalar_missing_source(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(rank, xs):
                return torch.tensor(xs[rank], device=self.rank)

            xs = []
            for i in range(self.world_size):
                xs.append(10 + i)

            f(self.rank, xs)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_type_mismatch(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x):
                if isinstance(x, int):
                    return torch.tensor(x, device=self.rank)
                else:
                    return x.sum()

            if self.rank == 0:
                x = torch.randn(10, device=self.rank)
            else:
                x = 12
            f(x)

            # This deadlocks, I guess we don't support this
            """
            if self.rank == 0:
                x = torch.randn(12, device=self.rank)
            else:
                x = 10
            f(x)
            """

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @enable_guard_collectives()
    def test_guard_collective(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x):
                return x.sum()

            x = torch.randn(10, device=self.rank)
            f(x)

            if self.rank == 0:
                x = torch.randn(10, device=self.rank)
            else:
                x = torch.randn(12, device=self.rank)  # recompile on one rank
            f(x)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._dynamo.config, "enable_compiler_collectives", True)
    @patch.object(torch._inductor.config, "max_autotune_gemm", True)
    @patch.object(torch._inductor.config, "distributed_max_autotune_gemm", True)
    def test_multiproc_autotune(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(a, b, c):
                res = (
                    torch.sum((a @ b) + 1.0)
                    + torch.sum(torch.relu(b @ c))
                    + torch.sum(c @ a)
                )

                return res

            a = torch.randn(1024, 1024, device=self.rank, dtype=torch.bfloat16)
            b = torch.randn(1024, 2048, device=self.rank, dtype=torch.bfloat16)
            c = torch.randn(2048, 1024, device=self.rank, dtype=torch.bfloat16)

            try:
                f(a, b, c)
            except Exception:
                log.exception("Caught exception running f")
                raise

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

            print(f"Result from {self.rank} is {f(a, b, c)}")

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._dynamo.config, "enable_compiler_collectives", True)
    @patch.object(torch._inductor.config, "max_autotune_gemm", True)
    @patch.object(torch._inductor.config, "distributed_max_autotune_gemm", True)
    def test_multiproc_autotune_dynamic_shapes(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(a, b, c):
                res = (
                    torch.sum((a @ b) + 1.0)
                    + torch.sum(torch.relu(b @ c))
                    + torch.sum(c @ a)
                )

                return res

            a = torch.randn(1024, 1024, device=self.rank, dtype=torch.bfloat16)
            b = torch.randn(1024, 2048, device=self.rank, dtype=torch.bfloat16)
            c = torch.randn(2048, 1024, device=self.rank, dtype=torch.bfloat16)

            # Mark tensors as dynamic on dimension 0
            torch._dynamo.mark_dynamic(a, 0)
            torch._dynamo.mark_dynamic(a, 1)
            torch._dynamo.mark_dynamic(b, 0)
            torch._dynamo.mark_dynamic(b, 1)
            torch._dynamo.mark_dynamic(c, 0)
            torch._dynamo.mark_dynamic(c, 1)

            try:
                f(a, b, c)
            except Exception:
                log.exception("Caught exception running f")
                raise

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

            print(f"Result from {self.rank} is {f(a, b, c)}")

            # Store the initial compilation count
            initial_compile_count = len(metrics)

            # # Test with different sizes to ensure dynamic shapes work without recompilation
            a2 = torch.randn(512, 512, device=self.rank, dtype=torch.bfloat16)
            b2 = torch.randn(512, 2048, device=self.rank, dtype=torch.bfloat16)
            c2 = torch.randn(2048, 512, device=self.rank, dtype=torch.bfloat16)

            try:
                result2 = f(a2, b2, c2)
                print(f"Result2 from {self.rank} is {result2}")
            except Exception:
                log.exception("Caught exception running f with different sizes")
                raise

            # Verify no recompilation occurred
            metrics_after = torch._dynamo.utils.get_compilation_metrics()
            final_compile_count = len(metrics_after)
            self.assertEqual(
                initial_compile_count,
                final_compile_count,
                "Expected no recompilation with dynamic shapes",
            )

            # Verify all ran
```



## High-Level Overview


This Python file contains 37 class(es) and 177 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ToyModel`, `MutatingModel`, `ForcedGetAttrMod`, `ToyInnerModel`, `ToyOuterModel`, `MyCustomLinear`, `MyLinear`, `MyModule`, `CheckSplitsCompiler`, `FakeDDP`, `TestFakeDistributedSingleProc`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `Model`, `TestMultiProc`

**Functions defined**: `reset_rng_state`, `init_weights`, `enable_guard_collectives`, `__init__`, `forward`, `get_model`, `__init__`, `forward`, `get_mutating_model`, `__init__`, `forward`, `get_forced_getattr_module`, `__init__`, `forward`, `__init__`, `forward`, `get_toy_model_for_activation_checkpointing`, `find_first_node`, `apply_fsdp_with_checkpointing`, `get_custom_model`

**Key imports**: contextlib, copy, functools, logging, random, unittest, contextmanager, timedelta, StringIO, patch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `copy`
- `functools`
- `logging`
- `random`
- `unittest`
- `datetime`: timedelta
- `io`: StringIO
- `unittest.mock`: patch
- `numpy as np`
- `torch`
- `torch._dynamo`
- `torch._dynamo.logging`
- `torch._dynamo.test_case`
- `torch.distributed as dist`
- `torch.optim as optim`
- `torch._C`: FileCheck
- `torch._dynamo.backends.distributed`: DDPOptimizer
- `torch._dynamo.comptime`: comptime
- `torch._dynamo.testing`: collect_results
- `torch._dynamo.utils`: same
- `torch._higher_order_ops.wrap`: tag_activation_checkpoint
- `torch.compiler`: set_enable_guard_collectives
- `torch.distributed._functional_collectives`: _maybe_wrap_tensor
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.nn.attention.flex_attention`: flex_attention
- `torch.nn.parallel`: DistributedDataParallel as DDP


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

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/test_dynamo_distributed.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed`):

- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`test_c10d_logger.py_docs.md`](./test_c10d_logger.py_docs.md)
- [`test_dist2.py_docs.md`](./test_dist2.py_docs.md)
- [`test_c10d_functional_native.py_docs.md`](./test_c10d_functional_native.py_docs.md)
- [`test_c10d_object_collectives.py_docs.md`](./test_c10d_object_collectives.py_docs.md)
- [`test_c10d_spawn_ucc.py_docs.md`](./test_c10d_spawn_ucc.py_docs.md)
- [`test_c10d_ucc.py_docs.md`](./test_c10d_ucc.py_docs.md)
- [`test_serialization.py_docs.md`](./test_serialization.py_docs.md)
- [`test_nccl.py_docs.md`](./test_nccl.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_dynamo_distributed.py_docs.md`
- **Keyword Index**: `test_dynamo_distributed.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
