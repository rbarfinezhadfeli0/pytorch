# Documentation: `test/distributed/test_c10d_common.py`

## File Metadata

- **Path**: `test/distributed/test_c10d_common.py`
- **Size**: 82,995 bytes (81.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import copy
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from sys import platform
from typing import Optional

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.distributed.distributed_c10d as c10d
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_SANDCASTLE,
    load_tests,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
    TestCase,
)
from torch.utils.checkpoint import checkpoint


if TEST_WITH_DEV_DBG_ASAN:
    print("Multiprocessing spawn is not compatible with dev/dbg asan", file=sys.stderr)
    sys.exit(0)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

if platform == "darwin":
    LOOPBACK = "lo0"
else:
    LOOPBACK = "lo"

torch.backends.cuda.matmul.allow_tf32 = False

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def gpus_for_rank(world_size):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    device_count = torch.accelerator.device_count()
    visible_devices = list(range(device_count))
    gpus_per_process = device_count // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank


class AbstractTimeoutTest:
    def _test_store_timeout(self, backend, init_method, c2p):
        try:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=1,
                rank=0,
                timeout=timedelta(seconds=1),
            )
            default_store = c10d._get_default_store()
            tik = time.time()
            with self.assertRaisesRegex(RuntimeError, "(?i)timeout"):
                default_store.get("nonexistent key")
            tok = time.time()
            dist.destroy_process_group()
            c2p.append(float(tok - tik))
        except RuntimeError as e:
            # catch "Address already in use" error and report it to the main
            # thread
            c2p.append(e)

    def _init_methods(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        if sys.platform == "win32":
            yield "file:///{}".format(f.name.replace("\\", "/"))
            f.close()
        else:
            yield f"file://{f.name}"
            f.close()
            yield f"tcp://127.0.0.1:{common.find_free_port():d}"

    def _test_default_store_timeout(self, backend):
        for init_method in self._init_methods():
            c2p = []
            t = threading.Thread(
                target=self._test_store_timeout, args=(backend, init_method, c2p)
            )
            t.daemon = True
            t.start()
            t.join(5)

            self.assertEqual(1, len(c2p))
            if isinstance(c2p[0], float):
                # waiting time should be 1s, use 3s to rule out false alarm
                self.assertGreater(3, c2p[0])
            elif isinstance(c2p[0], RuntimeError):
                # let @retry_on_connect_failures handle the error
                raise c2p[0]
            else:
                raise RuntimeError(f"Unexpected type {type(c2p[0])}")


class TimeoutTest(TestCase):
    @retry_on_connect_failures
    def test_store_based_barrier(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        port = common.find_free_port()

        def thread_work(timeout, init_type, world_size, rank, error_list):
            # we need to create a separate store just for the store barrier test
            if init_type == "file":
                barrier_store = dist.FileStore(f.name)
            elif init_type == "tcp":
                barrier_store = dist.TCPStore(
                    "localhost",
                    port,
                    world_size,
                    is_master=rank == 0,
                    wait_for_workers=False,
                )
            elif init_type == "hash":
                barrier_store = dist.HashStore()
            try:
                # 1 missing worker will cause it to timeout
                if rank != world_size - 1:
                    c10d._store_based_barrier(
                        rank=rank,
                        store=barrier_store,
                        group_name="_",
                        rendezvous_count=world_size,
                        timeout=timeout,
                        logging_interval=timeout / 2,
                    )
            except torch.distributed.DistStoreError as e:
                self.assertTrue(isinstance(e, torch.distributed.DistError))
                error_list.append(e)

        world_size = 4
        error_list = []
        threads = []
        for init_type in ["file", "tcp", "hash"]:
            for rank in range(world_size):
                t = threading.Thread(
                    target=thread_work,
                    args=(
                        timedelta(seconds=3),
                        init_type,
                        world_size,
                        rank,
                        error_list,
                    ),
                )
                threads.append(t)
                t.start()

            for thread in threads:
                thread.join()

            # we expect the world_size-1 threads to have failed
            self.assertEqual(len(error_list), world_size - 1)
            for error in error_list:
                self.assertTrue(
                    "Timed out initializing process group in store based barrier"
                    in error.args[0]
                )
            error_list = []
            threads = []


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class DoubleGpuNet(nn.Module):
    def __init__(self, gpus):
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[1])
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        ).to(gpus[0])

    def forward(self, x):
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        x = self.relu(self.fc1(x.to(dev0)))
        x = self.relu(self.fc2(x.to(dev1)))
        x = self.fc3(x)
        return F.softmax(x, dim=1).to(dev0)


class QuadraGpuNet(nn.Module):
    def __init__(self, gpus):
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[2])
        self.fc4 = nn.Linear(4, 4, bias=False).to(gpus[3])
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        ).to(gpus[0])

    def forward(self, x):
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        dev2 = self.fc3.weight.device
        dev3 = self.fc4.weight.device
        x = self.relu(self.fc1(x.to(dev0)))
        x = self.relu(self.fc2(x.to(dev1)))
        x = self.relu(self.fc3(x.to(dev2)))
        x = self.fc4(x.to(dev3))
        return F.softmax(x, dim=1).to(dev0)


class ConvNet(nn.Module):
    def __init__(self, gpus, layouts, dtypes):
        super().__init__()
        self.dtypes = dtypes
        if isinstance(gpus, list):
            self.layer_gpus = gpus
        else:
            gpus = [gpus] * 4
        self.conv0 = torch.nn.Conv2d(8, 16, (2, 2)).to(
            device=gpus[0], memory_format=layouts[0], dtype=dtypes[0]
        )
        self.conv1 = torch.nn.Conv2d(16, 32, (2, 2)).to(
            device=gpus[1], memory_format=layouts[1], dtype=dtypes[1]
        )
        self.conv2 = torch.nn.Conv2d(32, 16, (2, 2)).to(
            device=gpus[2], memory_format=layouts[2], dtype=dtypes[2]
        )
        self.conv3 = torch.nn.Conv2d(16, 8, (2, 2)).to(
            device=gpus[3], memory_format=layouts[3], dtype=dtypes[3]
        )

    def forward(self, x):
        x = x.to(self.dtypes[0])
        # Could say
        # x = self.conv0(x).to(device=self.conv1.weight.device, dtype=self.dtypes[1])
        # etc.  But I don't want to appeal to the weights' devices directly, because part of this test's purpose
        # is to verify weights are where expected if the model gets replicated.
        gpus = self.layer_gpus if hasattr(self, "layer_gpus") else [x.device] * 4
        x = self.conv0(x).to(device=gpus[1], dtype=self.dtypes[1])
        x = self.conv1(x).to(device=gpus[2], dtype=self.dtypes[2])
        x = self.conv2(x).to(device=gpus[3], dtype=self.dtypes[3])
        return self.conv3(x)


# A model involving FFTs, used to test DDP with complex tensors
class FFTModel(nn.Module):
    def __init__(self, hin, win, n_features):
        super().__init__()
        self.hin = hin
        self.win = win
        self.weight = nn.Parameter(
            torch.ones((n_features, n_features, hin, win // 2 + 1), dtype=torch.cfloat)
        )

    def forward(self, x):
        xc = torch.fft.rfft2(x, s=(self.hin, self.win), dim=(-2, -1), norm="ortho")
        xcw = torch.einsum("nchw,cohw->nohw", xc, self.weight)
        x = torch.fft.irfft2(xcw, dim=(-2, -1), norm="ortho")
        return x


class Task(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        return self.p + x


class ModuleForDdpCommHook(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t0 = Task()

    def forward(self, x, rank):
        return self.t0(x + rank)


class SparseGradientModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(10, 10, sparse=True)

    def forward(self, x):
        return F.softmax(self.embedding(x), dim=1)


class CommonDistributedDataParallelTest:
    def tearDown(self):
        # DistributedDataParallel test doesn't seem to call FileStore destructor
        # TODO: investigate this test and the test is known to have issues
        # Use this hack to remove files for that test
        try:
            os.remove(self.file_name)
        except (OSError, AttributeError):
            pass

    @property
    def world_size(self):
        return 2

    def _prepare_single_device_module(
        self,
        process_group,
        devices,
        device_ids,
        global_batch_size,
        gradient_as_bucket_view=False,
    ):
        model = Net()
        device = devices[0] if devices else torch.device(f"cuda:{self.rank:d}")
        ddp_model = DistributedDataParallel(
            copy.deepcopy(model).to(device),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        model.to(device)

        input = torch.randn(global_batch_size, 2).to(device)
        target = torch.randn(global_batch_size, 4).to(device)

        return model, ddp_model, input, target

    def _prepare_multi_device_module(
        self,
        process_group,
        devices,
        device_ids,
        global_batch_size,
        gradient_as_bucket_view=False,
    ):
        self.assertTrue(
            len(devices) == 2 or len(devices) == 4,
            f"unexpected devices for ddp tests {devices}",
        )
        if len(devices) == 2:
            model = DoubleGpuNet(devices)
        elif len(devices) == 4:
            model = QuadraGpuNet(devices)

        ddp_model = DistributedDataParallel(
            copy.deepcopy(model),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        input = torch.randn(global_batch_size, 2).to(devices[0])
        target = torch.randn(global_batch_size, 4)

        return model, ddp_model, input, target

    def _get_store(self):
        return dist.FileStore(self.file_name, self.world_size)

    def _get_process_group(self):
        raise NotImplementedError("To be implemented by child class")

    def _train_model(
        self, model, input_var, target, loss, run_checkpoint=False, use_reentrant=True
    ):
        model.train()
        if run_checkpoint:
            output = checkpoint(model, input_var, use_reentrant=use_reentrant)
        else:
            output = model(input_var)
        l = loss(output, target)
        l.backward()

    def _test_ddp_checkpointing(
        self,
        input_model,
        process_group,
        use_bucket_view,
        find_unused_parameters=False,
        static_graph=False,
        run_checkpoint=False,
        use_reentrant=True,
        allow_none_grads=False,
    ):
        # to reproduce the same training results
        torch.accelerator.set_device_index(self.rank)
        torch.manual_seed(31415)
        model = copy.deepcopy(input_model).to(device_type)
        ddp_model = copy.deepcopy(input_model).to(device_type)
        ddp_model = nn.parallel.DistributedDataParallel(
            ddp_model,
            bucket_cap_mb=1,
            gradient_as_bucket_view=use_bucket_view,
            device_ids=[self.rank],
            process_group=process_group,
            find_unused_parameters=find_unused_parameters,
            static_graph=static_graph,
        )
        self.assertEqual(
            ddp_model._get_ddp_logging_data().get("static_graph", 0), static_graph
        )
        input, ddp_input, target, ddp_target = self._prepare_dummy_data()
        loss = nn.MSELoss()
        n_iters = 5
        for i in range(n_iters):
            model.zero_grad(set_to_none=False)
            ddp_model.zero_grad(set_to_none=False)
            self._train_model(
                model,
                input,
                target,
                loss,
                run_checkpoint=run_checkpoint,
                use_reentrant=use_reentrant,
            )
            self._train_model(
                ddp_model,
                ddp_input,
                ddp_target,
                loss,
                run_checkpoint=run_checkpoint,
                use_reentrant=use_reentrant,
            )
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                if not allow_none_grads:
                    self.assertTrue(i.grad is not None)
                    self.assertTrue(j.grad is not None)
                self.assertEqual(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)

    # A list of tests for ddp with activation checkpointing
    # when gradient_as_bucket_view=True, False.
    # Most of the tests are referred to
    # https://github.com/facebookresearch/fairscale/blob/main/tests/nn/pipe/test_checkpoint_ddp.py
    class CheckpointOnceModule(nn.Module):
        """
        Runs checkpoint for a single layer in the model.
        """

        def __init__(self, use_reentrant=True):
            super().__init__()
            self.l1 = nn.Linear(20, 20)
            self.l2 = nn.Linear(20, 20)
            self.use_reentrant = use_reentrant

        def forward(self, inp):
            x = self.l1(inp)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            return x

    class CheckpointTwiceModule(CheckpointOnceModule):
        """
        Runs checkpoint for the same layer twice in a model. This simulates use
        cases such as pipeline parallel where the same layer can be checkpointed
        more than one time.
        """

        def __init__(self, use_reentrant=True):
            super().__init__(use_reentrant=use_reentrant)

        def forward(self, inp):
            x = self.l1(inp)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            return x

    class CheckpointTwiceModuleWeightSharing(CheckpointTwiceModule):
        """
        Similar to CheckpointTwiceModule but the weights are shared.
        """

        def __init__(self, use_reentrant=True):
            super().__init__(use_reentrant=use_reentrant)
            # Share weights
            self.l1.weight = self.l2.weight

        def forward(self, inp):
            x = self.l1(inp)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            return x

    class DynamicCheckpointTwiceModule(CheckpointTwiceModule):
        def __init__(self, use_reentrant=True):
            super().__init__(use_reentrant=use_reentrant)
            self.count = 0

        def forward(self, inp):
            if self.count % 2:
                x = checkpoint(self.l1, inp, use_reentrant=self.use_reentrant)
            else:
                x = checkpoint(self.l2, inp, use_reentrant=self.use_reentrant)

            self.count += 1
            return x

    class DynamicCheckpointTwiceModuleWeightSharing(DynamicCheckpointTwiceModule):
        def __init__(self, use_reentrant=True):
            super().__init__(use_reentrant=use_reentrant)
            # Share weights
            self.l1.weight = self.l2.weight

    def _prepare_dummy_data(self):
        ddp_bs = 16
        bs = ddp_bs * self.world_size
        input = torch.rand((bs, 20), device=device_type, requires_grad=True)
        target = torch.randn((bs, 20), device=device_type)
        offset = self.rank * ddp_bs
        ddp_input = input[offset : offset + ddp_bs]
        ddp_target = target[offset : offset + ddp_bs]
        return input, ddp_input, target, ddp_target

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_once(self, use_reentrant):
        """
        DDP works as expected when layer is checkpointed only once.
        """
        process_group = self._get_process_group()
        for use_bucket_view, static_graph in product((False, True), (False, True)):
            self._test_ddp_checkpointing(
                self.CheckpointOnceModule(use_reentrant=use_reentrant),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=static_graph,
            )
            if static_graph:
                # find_unused_parameters does not make a difference, since it is
                # ignored for static graph.
                self._test_ddp_checkpointing(
                    self.CheckpointOnceModule(),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    static_graph=static_graph,
                    find_unused_parameters=True,
                )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_unused_params(self, use_reentrant):
        """
        With reentrant autograd checkpointing impl, DDP will fail when there are
        unused params in the model and no static graph training. With
        non-reentrant checkpointing implementation, this works as expected.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            err_ctx = (
                nullcontext()
                if not use_reentrant
                else self.assertRaisesRegex(
                    RuntimeError, "Expected to mark a variable ready only once."
                )
            )
            with err_ctx:
                self._test_ddp_checkpointing(
                    self.CheckpointOnceModule(use_reentrant=use_reentrant),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    find_unused_parameters=True,
                )
            # test passes when static_graph is true
            self._test_ddp_checkpointing(
                self.CheckpointOnceModule(use_reentrant=use_reentrant),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                find_unused_parameters=True,
                static_graph=True,
            )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_twice(self, use_reentrant):
        """
        Checkpointing twice fails for non-static graph with reentrant checkpoint
        implementation, succeeds with non-reentrant checkpoint implementation.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            err_ctx = (
                nullcontext()
                if not use_reentrant
                else self.assertRaisesRegex(
                    RuntimeError, "Expected to mark a variable ready only once."
                )
            )
            with err_ctx:
                self._test_ddp_checkpointing(
                    self.CheckpointTwiceModule(use_reentrant=use_reentrant),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    static_graph=False,
                )

            with err_ctx:
                self._test_ddp_checkpointing(
                    self.CheckpointTwiceModule(use_reentrant=use_reentrant),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    static_graph=False,
                    find_unused_parameters=True,
                )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_twice_static_graph(self, use_reentrant):
        """
        Regardless of reentrant or non-reentrant checkpointing impl,
        checkpointing twice works with static graph enabled.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            # Test passes when static_graph=True.
            self._test_ddp_checkpointing(
                self.CheckpointTwiceModule(use_reentrant=use_reentrant),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=True,
            )

    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_dynamic_module(self):
        """
        Dynamic module can be checkpointed, multiple times, with non-reentrant
        checkpointing implementation.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            self._test_ddp_checkpointing(
                self.DynamicCheckpointTwiceModule(use_reentrant=False),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=False,
                find_unused_parameters=True,
                # Grads can be none sometimes due to dynamic module not using
                # all params.
                allow_none_grads=True,
            )

    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_dynamic_weight_sharing(self):
        """
        Dynamic module can be checkpointed multiple times with weight sharing
        using non-reentrant checkpointing implementation.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            self._test_ddp_checkpointing(
                self.DynamicCheckpointTwiceModuleWeightSharing(use_reentrant=False),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=False,
                find_unused_parameters=True,
                # Grads can be none sometimes due to dynamic module not using
                # all params.
                allow_none_grads=True,
            )

    # DDP works as expected if there is weight sharing among layers
    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_weight_sharing(self, use_reentrant):
        """
        Test that checkpointing with weight sharing works.
        """
        process_group = self._get_process_group()
        torch.accelerator.set_device_index(self.rank)
        for use_bucket_view, static_graph in product((False, True), (False, True)):
            torch.manual_seed(31415)
            l1 = nn.Linear(20, 20)
            l2 = nn.Linear(20, 20)
            l1.weight = l2.weight
            model = nn.Sequential(l1, l2)
            self._test_ddp_checkpointing(
                model,
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=static_graph,
                run_checkpoint=True,
                use_reentrant=use_reentrant,
            )

    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_twice_weight_sharing(self):
        """
        Checkpointing should work with static graph in the case of checkpointing
        same layer twice and having weights shared across layers.
        """
        process_group = self._get_process_group()
        torch.accelerator.set_device_index(self.rank)
        for use_bucket_view in (True, False):
            self._test_ddp_checkpointing(
                self.CheckpointTwiceModuleWeightSharing(),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=True,
            )

    def test_invalid_powerSGD_state(self):
        for start_powerSGD_iter, use_error_feedback, warm_start in product(
            [0, 1], [True, False], [True, False]
        ):
            if not use_error_feedback and not warm_start:
                continue
            with self.assertRaisesRegex(
                ValueError,
                "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                "because PowerSGD can only be applied after the first two iterations in DDP.",
            ):
                powerSGD.PowerSGDState(
                    process_group=None,
                    matrix_approximation_rank=1,
                    start_powerSGD_iter=start_powerSGD_iter,
                    use_error_feedback=use_error_feedback,
                    warm_start=warm_start,
                )

    def _test_ddp_with_process_group(
        self,
        process_group,
        devices,
        device_ids,
        multi_device=False,
        gradient_as_bucket_view=False,
    ):
        """
        Note: we pass down `device_ids` all the way to DistributedDataParallel
        as part of the test. Below you find tests that either use a list of
        integers, a list of `torch.Device` instances, or an empty list.
        The `devices` argument is used to control placement of the model and
        must always be specified as list of `torch.Device` instances.
        """
        local_batch_size = 1 if devices is None else len(devices)
        global_batch_size = self.world_size * local_batch_size

        if multi_device:
            model, ddp_model, input, target = self._prepare_multi_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )
            ddp_logging_data = ddp_model._get_ddp_logging_data()
            self.assertTrue(ddp_logging_data.get("is_multi_device_module"))
        else:
            model, ddp_model, input, target = self._prepare_single_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )
            ddp_logging_data = ddp_model._get_ddp_logging_data()
            self.assertFalse(ddp_logging_data.get("is_multi_device_module"))

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        def update_parameters(model):
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        # check two model parameters over 2 iterations
        for iteration in range(2):
            # single cpu/gpu training
            step_model(model, input, target)

            # DDP training, DDP scatters subsets of input_cpu to nodes/GPUs
            step_model(
                ddp_model,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            # Update weights and run a second iteration to shake out errors
            update_parameters(model)
            update_parameters(ddp_model)
            self.assertEqual(
                len(list(model.parameters())), len(list(ddp_model.parameters()))
            )
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    def _gpu_model_with_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False, state=None
    ):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Register a DDP communication hook if any.
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    def _gpu_model_with_builtin_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False
    ):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Register a built-in DDP communication hook if defined
        if hook is not None:
            gpu_model._register_builtin_comm_hook(hook)

        return gpu_model

    def _run_and_verify_hook(self, model, input, expected_grad):
        # Run forward
        output = model(input, self.rank)

        # Run backward
        output.mean().backward()

        [self.assertEqual(p.grad, expected_grad) for p in model.parameters()]

    def _simple_hook(
        self, state: object, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        fut = torch.futures.Future()
        fut.set_result(torch.ones_like(bucket.buffer()))

        def fut_then(fut):
            # Add ones to fut's result.
            t = fut.value()
            return t + torch.ones_like(t)

        return fut.then(fut_then)

    def _test_not_nan(self, model, x):
        y = model(x)
        self.assertFalse(y.isnan().any().item())
        y.sum().backward()
        for p in model.parameters():
            self.assertFalse(p.grad.isnan().any().item())

    @skip_if_lt_x_gpu(2)
    def test_sync_batch_norm_only_empty_input(self):
        pg = self._get_process_group()

        model = torch.nn.Sequential(
            nn.BatchNorm2d(2),
        ).to(device=self.rank)
        model = DistributedDataParallel(
            model,
            device_ids=[self.rank],
            process_group=pg,
        )
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            model,
            process_group=pg,
        )

        model.train()

        # only rank 0 receives empty inputs
        x = torch.zeros(
            (1 if self.rank != 0 else 0, 2, 11, 13),
            dtype=torch.float32,
            device=self.rank,
        )

        # input requires grad, this will trigger the collective communication
        # in the backward pass
        x.requires_grad = True
        self._test_not_nan(model, x)

        # input does not requires grad
        x.requires_grad = False
        self._test_not_nan(model, x)

        # all ranks receive empty inputs
        x = torch.zeros((0, 2, 11, 13), dtype=torch.float32, device=self.rank)

        # input requires grad, this will trigger the collective communication
        # in the backward pass
        x.requires_grad = True
        self._test_not_nan(model, x)

        # input does not requires grad
        x.requires_grad = False
        self._test_not_nan(model, x)

    @skip_if_lt_x_gpu(2)
    def test_sync_batch_norm_empty_input(self):
        pg = self._get_process_group()

        model = torch.nn.Sequential(
            nn.Conv2d(2, 2, 3),
            nn.BatchNorm2d(2),
            nn.Linear(28, 2),
        ).to(device=self.rank)
        model = DistributedDataParallel(
            model,
            device_ids=[self.rank],
            process_group=pg,
        )
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            model,
            process_group=pg,
        )

        model.train()
        # only rank 0 receives empty inputs
        x = torch.zeros(
            (3 if self.rank != 0 else 0, 2, 30, 30),
            dtype=torch.float32,
            device=self.rank,
        )

        self._test_not_nan(model, x)

        # all ranks receive empty inputs
        x = torch.zeros((0, 2, 30, 30), dtype=torch.float32, device=self.rank)

        self._test_not_nan(model, x)

    @dataclass
    class CustomOutput:
        o1: Optional[torch.Tensor]
        o2: dict[str, torch.Tensor]

    class DataclassOutputModule(nn.Module):
        def __init__(self, skip_o1):
            super().__init__()
            self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(3)])
            self.relu = nn.ReLU()
            self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(3)])
            self.skip_o1 = skip_o1

        def forward(self, x):
            o1 = None if self.skip_o1 else self.relu(self.seq1(x))
            o2 = {"a": self.seq2(x), "b": self.relu(self.seq2(x))}
            return CommonDistributedDataParallelTest.CustomOutput(o1=o1, o2=o2)

    def _test_dataclass_output(self, skip_o1):
        net_x = torch.cat([torch.ones(4, 10) * i for i in range(self.world_size)]).to(
            self.rank
        )
        ddp_x = torch.ones(4, 10, device=self.rank) * self.rank

        # use manual_seed to make sure local models start with the same values
        torch.manual_seed(0)
        net = self.DataclassOutputModule(skip_o1=skip_o1).to(self.rank)
        ddp = DistributedDataParallel(
            copy.deepcopy(net),
            device_ids=[self.rank],
            find_unused_parameters=True,
            static_graph=False,
            process_group=self._get_process_group(),
        )

        net_out = net(net_x)
        ddp_out = ddp(ddp_x)

        net_loss = F.mse_loss(
            (
                net_out.o1 + net_out.o2["a"] + net_out.o2["b"]
                if not skip_o1
                else net_out.o2["a"] + net_out.o2["b"]
            ),
            torch.ones_like(net_out.o2["a"], device=self.rank),
        )
        ddp_loss = F.mse_loss(
            (
                ddp_out.o1 + ddp_out.o2["a"] + ddp_out.o2["b"]
                if not skip_o1
                else ddp_out.o2["a"] + ddp_out.o2["b"]
            ),
            torch.ones_like(ddp_out.o2["a"], device=self.rank),
        )

        net_loss.backward()
        ddp_loss.backward()

        for p1, p2 in zip(net.parameters(), ddp.parameters()):
            if torch.is_tensor(p1.grad):
                self.assertTrue(p1.grad.allclose(p2.grad))
            else:
                self.assertEqual(p1.grad, p2.grad)

    @skip_if_lt_x_gpu(2)
    def test_dataclass_output(self):
        self._test_dataclass_output(skip_o1=False)

    @skip_if_lt_x_gpu(2)
    def test_dataclass_output_unused_param(self):
        self._test_dataclass_output(skip_o1=True)


class ComputeBucketAssignmentTest(TestCase):
    def test_single_limit_single_dtype(self):
        tensors = [
            torch.empty([100], dtype=torch.float),
            torch.empty([200], dtype=torch.float),
            torch.empty([100], dtype=torch.float),
            torch.empty([50], dtype=torch.float),
        ]
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [400]
        )
        self.assertTrue(all(size_lim == 400 for size_lim in per_bucket_size_limits))
        self.assertEqual([[0], [1], [2], [3]], result)

    def test_single_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [400]
        )
        self.assertTrue(all(size_lim == 400 for size_lim in per_bucket_size_limits))
        self.assertEqual([[0, 2], [1, 3], [4], [5]], result)

    def test_multi_limit_single_dtype(self):
        tensors = [
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
        ]
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [40, 80]
        )
        self.assertEqual(per_bucket_size_limits, [40, 80, 80])
        self.assertEqual([[0], [1, 2], [3]], result)

    def test_multi_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [200, 400]
        )
        self.assertEqual([[0], [1], [2, 4], [3, 5]], result)
        self.assertEqual(per_bucket_size_limits, [200, 200, 400, 400])


class AbstractCommTest:
    @property
    def op_timeout_sec(self):
        return 1

    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        self.fail("test subclass didn't override device")

    def _verify_sequence_number_across_pg(self, pg, verify_pg):
        seq_num = pg._get_sequence_number_for_group()
        obj_list = [None for _ in range(dist.get_world_size(verify_pg))]
        # We use a separate pg to verify the sequence numbers, otherwise these
        # collectives will themselves increment the sequence number.
        dist.all_gather_object(obj_list, seq_num, group=verify_pg)
        self.assertEqual(len(set(obj_list)), 1)
        return obj_list[0]

    def _test_sequence_num_incremented(self, process_group, ranks):
        # verify initial sequence numbers. Use a distinct process group for
        # verification to keep counts as expected with respect to process_group.
        verify_pg = dist.new_group(
            ranks=ranks,
            backend="gloo",
        )
        assert dist.get_world_size(process_group) == dist.get_world_size(verify_pg)

        initial_num = (
            self._verify_sequence_number_across_pg(
                pg=process_group, verify_pg=verify_pg
            )
            if not c10d._rank_not_in_group(process_group)
            else -1
        )

        # Verify sequence numbers are appropriately incremented
        for i in range(10):
            t = torch.ones(1, device=device_type)
            dist.all_reduce(t, group=process_group)
            if not c10d._rank_not_in_group(process_group):
                seq_num = self._verify_sequence_number_across_pg(
                    pg=process_group,
                    verify_pg=verify_pg,
                )
                self.assertEqual(initial_num + i + 1, seq_num)

        if dist.get_world_size(process_group) > 2:
            # Test when certain ranks don't call collectives
            if dist.get_rank(process_group) not in [0, 2]:
                dist.all_reduce(t, group=process_group, async_op=True)
            # Now ranks 0 and 2 should be lagging by 1.
            if not c10d._rank_not_in_group(process_group):
                seq_num = process_group._get_sequence_number_for_group()
                rank = dist.get_rank(process_group)
                obj_list = [None for _ in range(dist.get_world_size(verify_pg))]
                dist.all_gather_object(obj_list, (rank, seq_num), group=verify_pg)
                rank_to_seq_num = dict(obj_list)
                self.assertEqual(len(set(rank_to_seq_num.values())), 2)
                self.assertEqual(rank_to_seq_num[0], rank_to_seq_num[2])
                expected_same = {
                    rank_to_seq_num[i] for i in rank_to_seq_num if i not in [0, 2]
                }
                self.assertEqual(len(expected_same), 1)
                self.assertEqual(rank_to_seq_num[0] + 1, rank_to_seq_num[1])

    def _test_sequence_num_incremented_default_group(self, backend_name):
        torch.accelerator.set_device_index(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        self._test_sequence_num_incremented(
            c10d._get_default_group(),
            ranks=list(range(dist.get_world_size())),
        )

    def _test_sequence_num_incremented_subgroup(self, backend_name):
        torch.accelerator.set_device_index(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        subgroup_ranks = [0, 1, 2]
        subgroup = dist.new_group(subgroup_ranks)
        self._test_sequence_num_incremented(subgroup, subgroup_ranks)

    def _test_sequence_num_set_default_pg(self, backend):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        default_pg = c10d._get_default_group()
        seq_num = default_pg._get_sequence_number_for_group()
        obj_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(obj_list, seq_num)
        self.assertEqual(len(set(obj_list)), 1)

    def _test_sequence_num_set_new_group(self, backend):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        subgroup = dist.new_group([0, 1])

        if not c10d._rank_not_in_group(subgroup):
            subgroup_seq = subgroup._get_sequence_number_for_group()
            obj_list = [None for _ in range(dist.get_world_size(subgroup))]
            dist.all_gather_object(obj_list, subgroup_seq, group=subgroup)
            self.assertEqual(len(set(obj_list)), 1)

    def _test_warn_not_in_group(self, backend):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        in_group_ranks = list(filter(lambda x: x % 2 == 0, range(self.world_size)))
        group = dist.new_group(in_group_ranks)

        x = torch.zeros(2, 2).to(self.rank)
        xs = [torch.zeros(2, 2).to(self.rank) for _ in range(len(in_group_ranks))]
        if self.rank not in in_group_ranks:
            msg = ".*{}.*does not belong to.*"
            with self.assertWarnsOnceRegex(UserWarning, msg.format("all_gather")):
                dist.all_gather(xs, x, group=group)
            with self.assertWarnsOnceRegex(UserWarning, msg.format("all_reduce")):
                dist.all_reduce(x, group=group)
            with self.assertWarnsOnceRegex(UserWarning, msg.format("barrier")):
                dist.barrier(group=group)
            with self.assertWarnsOnceRegex(UserWarning, msg.format("broadcast")):
                dist.broadcast(x, src=0, group=group)
        else:
            dist.all_gather(xs, x, group=group)
            dist.all_reduce(x, group=group)
            dist.barrier(group=group)
            dist.broadcast(x, src=0, group=group)

    def _test_rank_membership(self, backend):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        self.assertTrue(self.world_size > 1)

        group = dist.new_group(ranks=[1])
        self.assertEqual(dist.get_group_rank(group, 1), 0)
        with self.assertRaisesRegex(ValueError, "not part of group"):
            dist.get_group_rank(group, 0)
        with self.assertRaisesRegex(ValueError, "not registered"):
            dist.get_group_rank(DummyProcessGroup(self.rank, self.world_size), 0)

        self.assertEqual(dist.get_global_rank(group, 0), 1)
        with self.assertRaisesRegex(ValueError, "not part of group"):
            dist.get_global_rank(group, 1)
        with self.assertRaisesRegex(ValueError, "not registered"):
            dist.get_global_rank(DummyProcessGroup(self.rank, self.world_size), 0)

        self.assertEqual(dist.get_process_group_ranks(group), [1])

    def _test_tensor_dtype_mismatch(self, backend):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        tensor = torch.ones(2, 2, device=self.device) * 7
        tensor_h = tensor.half()
        tensor_list = [
            torch.zeros(2, 2, device=self.device) for _ in range(self.world_size)
        ]
        tensor_list_h = list(tensor_list)
        tensor_list_h[1] = tensor_list_h[1].half()

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_gather(tensor_list_h, tensor)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_gather(tensor_list, tensor_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_gather_coalesced([tensor_list_h], tensor_list)
            dist.all_gather_coalesced([tensor_list], tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_reduce_coalesced(tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.reduce_scatter(tensor, tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.reduce_scatter(tensor_h, tensor_list)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_to_all_single(tensor_h, tensor)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_to_all(tensor_list_h, tensor_list)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_to_all(tensor_list, tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.scatter(tensor, tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.gather(tensor_h, tensor_list)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.gather(tensor, tensor_list_h)

        with sel
```



## High-Level Overview


This Python file contains 31 class(es) and 151 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AbstractTimeoutTest`, `TimeoutTest`, `Net`, `DoubleGpuNet`, `QuadraGpuNet`, `ConvNet`, `FFTModel`, `Task`, `ModuleForDdpCommHook`, `SparseGradientModule`, `CommonDistributedDataParallelTest`, `CheckpointOnceModule`, `CheckpointTwiceModule`, `CheckpointTwiceModuleWeightSharing`, `DynamicCheckpointTwiceModule`, `DynamicCheckpointTwiceModuleWeightSharing`, `CustomOutput`, `DataclassOutputModule`, `ComputeBucketAssignmentTest`, `AbstractCommTest`

**Functions defined**: `gpus_for_rank`, `_test_store_timeout`, `_init_methods`, `_test_default_store_timeout`, `test_store_based_barrier`, `thread_work`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`

**Key imports**: copy, os, pickle, subprocess, sys, tempfile, threading, time, unittest, nullcontext


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `os`
- `pickle`
- `subprocess`
- `sys`
- `tempfile`
- `threading`
- `time`
- `unittest`
- `contextlib`: nullcontext
- `dataclasses`: dataclass
- `datetime`: timedelta
- `itertools`: product
- `typing`: Optional
- `torch`
- `torch.distributed as dist`
- `torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD`
- `torch.distributed.distributed_c10d as c10d`
- `torch.nn.functional as F`
- `torch.testing._internal.common_utils as common`
- `torch.nn.parallel`: DistributedDataParallel
- `torch.utils.checkpoint`: checkpoint
- `torch.distributed.distributed_c10d`: _canonicalize_group_rank


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/test_c10d_common.py
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

- **File Documentation**: `test_c10d_common.py_docs.md`
- **Keyword Index**: `test_c10d_common.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
