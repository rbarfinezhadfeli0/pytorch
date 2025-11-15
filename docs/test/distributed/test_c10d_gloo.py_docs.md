# Documentation: `test/distributed/test_c10d_gloo.py`

## File Metadata

- **Path**: `test/distributed/test_c10d_gloo.py`
- **Size**: 103,090 bytes (100.67 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import copy
import json
import logging
import math
import operator
import os
import pickle
import random
import sys
import tempfile
import time
from datetime import timedelta
from functools import reduce
from itertools import groupby

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_gloo_available():
    print("c10d GLOO not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import test_c10d_common
from test_c10d_common import (
    FFTModel,
    gpus_for_rank,
    LOOPBACK,
    ModuleForDdpCommHook,
    SparseGradientModule,
    Task,
)

import torch.distributed as dist
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardedTensor,
    ShardMetadata,
)
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    create_device,
    MultiProcessTestCase,
    requires_gloo,
    simple_sparse_reduce_tests,
    skip_if_lt_x_gpu,
    skip_if_win32,
    verify_ddp_error_logged,
)
from torch.testing._internal.common_utils import (
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle,
    skipIfRocm,
    TestCase,
)


def simple_reduce_tests(rank, world_size):
    tests = [
        (
            c10d.ReduceOp.SUM,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.tensor([rank + 1.0]),
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size)]),
        ),
        (
            c10d.ReduceOp.AVG,
            torch.tensor([rank + 1.0]),
            torch.tensor([float((world_size + 1) / 2)]),
        ),
    ]

    # Generate tests for BAND.
    # The bit that is set changes in every iteration to check
    # that the output changes accordingly.
    for i in range(4):
        vin = rank | (1 << i)
        vout = 1 << i
        tests.append(
            (
                c10d.ReduceOp.BAND,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for BOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-OR'ed.
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])
        vout = reduce(operator.or_, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for XOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-XOR'ed.
    for i in range(1, 5):
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        vout = reduce(operator.xor, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BXOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Extend tests for cfloat dtype
    tests.extend(
        (
            (
                c10d.ReduceOp.SUM,
                torch.tensor([complex(rank + 1.0, rank + 1.0)], dtype=torch.cfloat),
                torch.tensor(
                    [
                        complex(
                            world_size * (world_size + 1) / 2,
                            world_size * (world_size + 1) / 2,
                        )
                    ],
                    dtype=torch.cfloat,
                ),
            ),
            (
                c10d.ReduceOp.AVG,
                torch.tensor([complex(rank + 1.0, rank + 1.0)], dtype=torch.cfloat),
                torch.tensor(
                    [complex(float((world_size + 1) / 2), float((world_size + 1) / 2))],
                    dtype=torch.cfloat,
                ),
            ),
        )
    )
    return tests


def simple_coalesced_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([rank + 1.0]), torch.tensor([(rank + 1.0) ** 2])],
            [
                torch.tensor([float(world_size * (world_size + 1) / 2)]),
                torch.tensor(
                    [float(world_size * (world_size + 1) * (2 * world_size + 1) / 6)]
                ),
            ],
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([rank + 1.0]), torch.tensor([rank + 2.0])],
            [
                torch.tensor([float(math.factorial(world_size))]),
                torch.tensor([float(math.factorial(world_size + 1))]),
            ],
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([rank + x]) for x in [0.0, 1.0]],
            [torch.tensor([0.0]), torch.tensor([1.0])],
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([rank + x]) for x in [1.0, 2.0]],
            [torch.tensor([float(world_size)]), torch.tensor([world_size + 1.0])],
        ),
    ]


def simple_multi_input_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([2 * rank + 0.0]), torch.tensor([2 * rank + 1.0])],
            torch.tensor([float(world_size * (2 * world_size - 1))]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([float(math.factorial(2 * world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([2.0 * world_size]),
        ),
    ]


class RendezvousTCPTest(TestCase):
    @retry_on_connect_failures
    def test_tcp_init(self):
        rendezvous_iterator = dist.rendezvous("tcp://127.0.0.1:0", rank=0, world_size=1)
        store, rank, world_size = next(rendezvous_iterator)
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)
        # port number should get assigned
        self.assertNotEqual(store.port, "0")


class RendezvousEnvTest(TestCase):
    @requires_gloo()
    @retry_on_connect_failures
    def test_logging_init(self):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"

        previous_handlers = logging.root.handlers

        c10d.init_process_group(backend="gloo", init_method="env://")

        current_handlers = logging.root.handlers
        self.assertEqual(len(previous_handlers), len(current_handlers))
        for current, previous in zip(current_handlers, previous_handlers):
            self.assertEqual(current, previous)

        c10d.destroy_process_group()


class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    @requires_gloo()
    @retry_on_connect_failures
    def test_default_store_timeout_gloo(self):
        self._test_default_store_timeout("gloo")


class ProcessGroupGlooTest(MultiProcessTestCase):
    lazy_init = False

    def _create_process_group_gloo(self, store, rank, world_size, opts):
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, opts)
        dist.barrier(group=pg)
        return pg

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def opts(self, threads=2, group_name="0"):
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 50.0
        opts._devices = [create_device(interface=LOOPBACK, lazy_init=self.lazy_init)]
        opts._threads = threads
        opts.group_name = group_name
        return opts

    @requires_gloo()
    def test_multi_device_constructor(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 5.0
        opts._devices = [
            create_device(interface=LOOPBACK, lazy_init=self.lazy_init),
            create_device(interface=LOOPBACK, lazy_init=self.lazy_init),
        ]
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, opts)

        # Execute 2x the number of operations to ensure we use every device.
        for fut in [pg.allreduce(torch.ones(i + 1)).get_future() for i in range(4)]:
            fut.wait()

    @requires_gloo()
    def test_empty_tensors(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        xs = [torch.FloatTensor([])]
        fut = pg.broadcast(xs).get_future()
        fut.wait()
        output = fut.value()
        self.assertEqual(0, output[0].numel())
        self.assertEqual(xs[0], output[0])

    @requires_gloo()
    def test_broadcast_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = -1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t2], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t3], opts)

    def _test_broadcast_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            fut = pg.broadcast(xs, opts).get_future()
            fut.wait()
            return fut.value()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.tensor([self.rank]))
            output = broadcast([x], i, 0)
            self.assertEqual(torch.tensor([i]), output[0])

            # Run with 2 input tensors
            num = 2
            for j in range(num):
                xs = [
                    fn(torch.tensor([self.rank * num + 0.0])),
                    fn(torch.tensor([self.rank * num + 1.0])),
                ]

                output = broadcast(xs, i, j)
                self.assertEqual(
                    torch.tensor([i * num + j], dtype=torch.float32), output[0]
                )
                self.assertEqual(
                    torch.tensor([i * num + j], dtype=torch.float32), output[1]
                )

            # Run with 1 input tensor of cfloat dtype
            x = fn(torch.tensor([complex(self.rank, self.rank)], dtype=torch.cfloat))
            output = broadcast([x], i, 0)
            self.assertEqual(
                torch.tensor([complex(i, i)], dtype=torch.cfloat), output[0]
            )

        # Test overloaded convenience function
        x = torch.tensor([self.rank + 1.0])
        fut = pg.broadcast(x, root=0).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([1.0]), result[0])

    @requires_gloo()
    def test_broadcast_basics(self):
        self._test_broadcast_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_broadcast_basics_cuda(self):
        self._test_broadcast_basics(lambda t: t.clone().cuda())

    def _test_broadcast_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        work_handles = [
            pg.broadcast(inputs[i], root=(i % self.world_size))
            for i in range(len(inputs))
        ]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.tensor([(i * self.world_size) + (i % self.world_size)]),
                inputs[i],
                msg=(f"Mismatch in iteration {i:d}"),
            )

    @requires_gloo()
    def test_broadcast_stress(self):
        inputs = [torch.tensor([i * self.world_size + self.rank]) for i in range(1000)]
        self._test_broadcast_stress(inputs)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_broadcast_stress_cuda(self):
        inputs = [
            torch.tensor([i * self.world_size + self.rank]).cuda() for i in range(1000)
        ]
        self._test_broadcast_stress(inputs)

    @requires_gloo()
    def test_allreduce_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t3], opts)

    @requires_gloo()
    def test_allreduce_op_timeout(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        opts = c10d.AllreduceOptions()
        opts.timeout = timedelta(milliseconds=1)

        if self.rank == 0:
            t1 = torch.zeros([1], dtype=torch.float32)
            with self.assertRaisesRegex(RuntimeError, "Timed out waiting 1ms"):
                pg.allreduce([t1], opts).wait()

    @requires_gloo()
    def test_allreduce_overall_timeout(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        pg.set_timeout(timedelta(milliseconds=1))

        if self.rank == 0:
            t1 = torch.zeros([1], dtype=torch.float32)
            with self.assertRaisesRegex(RuntimeError, "Timed out waiting 1ms"):
                pg.allreduce([t1]).wait()

    def _test_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for op, input, expected in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            result = fut.value()
            self.assertEqual(expected, result[0])

        # Multi input tests
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for op, inputs, output in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensors = [fn(input) for input in inputs]
            fut = pg.allreduce(tensors, opts).get_future()
            fut.wait()
            result = fut.value()
            for tensor in result:
                self.assertEqual(output, tensor)

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.tensor([self.rank + 1.0]))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]),
            result[0],
        )

        # Test fp16 numerical correctness for all-reduce SUM.
        torch.manual_seed(self.rank)
        # TODO: when create larger sizes of tensors, numerical instability will be observed.
        # We need to investigate the root cause and ensure it is fixed.
        tensor = (
            (torch.rand(200, 1, dtype=torch.float32) * 2 - 1) * 65504 / self.world_size
        )
        opts = c10d.AllreduceOptions()
        tensor = tensor.to(torch.float16)
        output = [[torch.zeros_like(tensor) for _ in range(self.world_size)]]
        # allgather all local tensors first and then sum up.
        fut = pg.allgather(output, [tensor]).get_future()
        fut.wait()
        ag_result = fut.value()
        total = torch.stack(ag_result, dim=0).sum(dim=0)

        # result from fp16 all-reduce.
        fut = pg.allreduce([tensor], opts).get_future()
        fut.wait()
        result_fp16 = fut.value()
        # float16 has only ~11 bits of mantissa, and is sensitive to accumulation
        # order and rounding errors so we use a larger tolerance.
        self.assertEqual(total, result_fp16[0], rtol=1e-2, atol=1e-3)

    @requires_gloo()
    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allreduce_basics_cuda(self):
        self._test_allreduce_basics(lambda t: t.clone().cuda())

    def _test_allreduce_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = [
            pg.allreduce(inputs[i]).get_future() for i in range(len(inputs))
        ]
        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            self.assertEqual(
                torch.tensor(
                    [
                        (i * self.world_size)
                        + (self.world_size * (self.world_size - 1) // 2)
                    ]
                ),
                future_handle.value()[0],
                msg=(f"Mismatch in iteration {i:d}"),
            )

    @requires_gloo()
    def test_allreduce_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_allreduce_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @requires_gloo()
    def test_allreduce_coalesced_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros(1, dtype=torch.float32)
        t2 = torch.zeros(1, dtype=torch.float64)
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(1,))

        with self.assertRaisesRegex(RuntimeError, "requires non-empty tensor list"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([], opts)

        with self.assertRaisesRegex(
            RuntimeError, "tensors must all have the same type"
        ):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t2], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor layout at index"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t3], opts)

        with self.assertRaisesRegex(RuntimeError, "unsupported layout"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t3, t3.clone()], opts)

    @skip_if_lt_x_gpu(1)
    @requires_gloo()
    def test_allreduce_coalesced_checks_cuda(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros(1, dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "unsupported device type"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1.cuda(), t1.cuda()], opts)

    def _test_allreduce_coalesced_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        test_cases = simple_coalesced_reduce_tests(self.rank, self.world_size)
        for op, inputs, outputs in test_cases:
            opts = c10d.AllreduceCoalescedOptions()
            opts.reduceOp = op
            tensors = [fn(x) for x in inputs]
            fut = pg.allreduce_coalesced(tensors, opts).get_future()
            fut.wait()
            result = fut.value()
            for result_tensor, expected in zip(result, outputs):
                self.assertEqual(result_tensor, expected)

    @requires_gloo()
    def test_allreduce_coalesced_basics(self):
        self._test_allreduce_coalesced_basics(lambda t: t.clone())

    def _expected_output(self, i):
        ws = self.world_size
        return 2 * [torch.tensor([(i * ws) + (ws * (ws - 1) // 2)])]

    def _test_allreduce_coalesced_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = [
            pg.allreduce_coalesced(input).get_future() for input in inputs
        ]
        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            self.assertEqual(
                self._expected_output(i),
                result,
                msg=f"Mismatch in iteration {i}",
            )

    @requires_gloo()
    def test_allreduce_coalesced_stress(self):
        inputs = [2 * [torch.tensor([i + self.rank])] for i in range(1000)]
        self._test_allreduce_coalesced_stress(inputs)

    @requires_gloo()
    def test_allreduce_coalesced_async(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )

        xs = [2 * [torch.tensor([i + self.rank])] for i in range(2)]
        futs = [c10d.all_reduce_coalesced(x, async_op=True) for x in xs]
        torch.futures.wait_all(futs)
        for i, fut in enumerate(futs):
            self.assertEqual(
                self._expected_output(i),
                fut.wait(),
                msg=f"Mismatch in iteration {i}",
            )

    @requires_gloo()
    def test_sparse_allreduce_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1])
        t2 = torch.sparse_coo_tensor([[0]], [1], size=(2,))
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(4,))

        with self.assertRaisesRegex(RuntimeError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor layout"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t2, t3], opts)

        # Sparse allreduce only works with c10d.ReduceOp.SUM.
        for op in [c10d.ReduceOp.PRODUCT, c10d.ReduceOp.MIN, c10d.ReduceOp.MAX]:
            with self.assertRaisesRegex(
                RuntimeError, "unsupported reduction operation"
            ):
                opts = c10d.AllreduceOptions()
                opts.reduceOp = op
                pg.allreduce([t3], opts)

    def _test_sparse_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        for num_inputs_per_rank in [1, 2]:
            tests = simple_sparse_reduce_tests(
                self.rank, self.world_size, num_inputs=num_inputs_per_rank
            )
            for inputs, outputs in tests:
                tensors = [fn(input) for input in inputs]
                fut = pg.allreduce(tensors).get_future()
                fut.wait()
                result = fut.value()
                self.assertEqual(tensors, outputs)
                self.assertEqual(result, outputs)

    @requires_gloo()
    def test_sparse_allreduce_basics(self):
        self._test_sparse_allreduce_basics(lambda t: t)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sparse_allreduce_basics_cuda(self):
        self._test_sparse_allreduce_basics(lambda t: t.clone().cuda())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sparse_allreduce_cuda_dispatched(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        tests = simple_sparse_reduce_tests(self.rank, self.world_size, num_inputs=1)
        for inputs, outputs in tests:
            tensors = inputs[-1].clone().cuda()
            work = dist.all_reduce(tensors, async_op=True)
            work.wait()
            self.assertEqual([tensors], outputs)

    @requires_gloo()
    def test_allgather_into_tensor_coalesced(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.manual_seed(42)
        in_shapes = [(5, 5), (10, 10), (15, 15)]
        out_shapes = [(s[0] * self.world_size,) + s[1:] for s in in_shapes]

        outputs = [torch.empty(s) for s in out_shapes]
        inputs = [torch.rand(s) for s in in_shapes]
        work = dist.group.WORLD.allgather_into_tensor_coalesced(outputs, inputs)
        work.wait()

        for output, input in zip(outputs, inputs):
            expect = torch.cat([input] * self.world_size)
            self.assertTrue(torch.allclose(output, expect))

    @requires_gloo()
    def test_reduce_scatter(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.manual_seed(42)

        # variable size per rank
        inputs = [torch.rand(i) for i in range(self.world_size)]
        output = torch.empty(self.rank)

        work = dist.reduce_scatter(output, inputs, async_op=True)
        work.wait()

        expect = inputs[self.rank] * self.world_size
        self.assertTrue(torch.allclose(output, expect))

    @requires_gloo()
    def test_reduce_scatter_tensor(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.manual_seed(42)
        out_shape = (20, 20)
        in_shape = (out_shape[0] * self.world_size,) + out_shape[1:]

        output = torch.empty(out_shape)
        input = torch.rand(in_shape)
        work = dist.reduce_scatter_tensor(output, input, async_op=True)
        work.wait()

        expect = (
            input.view(self.world_size, *out_shape).chunk(self.world_size)[self.rank]
            * self.world_size
        )
        self.assertTrue(torch.allclose(output, expect))

    @requires_gloo()
    def test_reduce_scatter_tensor_coalesced(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.manual_seed(42)
        out_shapes = [(5, 5), (10, 10), (15, 15)]
        in_shapes = [(s[0] * self.world_size,) + s[1:] for s in out_shapes]

        outputs = [torch.empty(s) for s in out_shapes]
        inputs = [torch.rand(s) for s in in_shapes]
        work = dist.group.WORLD.reduce_scatter_tensor_coalesced(outputs, inputs)
        work.wait()

        for output, input in zip(outputs, inputs):
            expect = (
                input.view(self.world_size, *output.shape).chunk(self.world_size)[
                    self.rank
                ]
                * self.world_size
            )
            self.assertTrue(torch.allclose(output, expect))

    @requires_gloo()
    def test_scatter_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element output tensor list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element output tensor list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([t1, t1], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element input list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element input list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * self.world_size, [t1] * self.world_size], opts)

        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = "Incorrect input list size {}. Input list size should be {}"
        with self.assertRaisesRegex(
            RuntimeError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)

        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(
            RuntimeError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t2] * self.world_size], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t3] * self.world_size], opts)

        with self.assertRaisesRegex(RuntimeError, "requires empty input on non-root"):
            opts = c10d.ScatterOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.scatter([t1], [[t1] * self.world_size], opts)

    def _test_scatter_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Preallocate tensors for input/output
        input = [fn(torch.tensor([self.rank])) for _ in range(self.world_size)]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the scatter root and accumulate work items
        futures = []
        for i in range(self.world_size):
            opts = c10d.ScatterOptions()
            opts.rootRank = i
            if i == self.rank:
                futures.append(pg.scatter([outputs[i]], [input], opts).get_future())
            else:
                futures.append(pg.scatter([outputs[i]], [], opts).get_future())

        # Wait for work to complete
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            self.assertEqual(torch.tensor([i]), result[0])

    @requires_gloo()
    def test_scatter_basics(self):
        self._test_scatter_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_scatter_basics_cuda(self):
        self._test_scatter_basics(lambda t: t.clone().cuda())

    def _test_scatter_stress(self, inputs, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        outputs = [
            [fn(torch.tensor([-1])) for _ in range(self.world_size)]
            for _ in range(len(inputs))
        ]
        future_handles = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ScatterOptions()
                opts.rootRank = root
                if root == self.rank:
                    fut = pg.scatter(
                        [outputs[i][root]], [[fn(e) for e in inputs[i]]], opts
                    ).get_future()
                else:
                    fut = pg.scatter([outputs[i][root]], [], opts).get_future()
                future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            result = future_handle.value()

            self.assertEqual(
                torch.tensor([iter + root]),
                result[0],
                msg=(f"Mismatch in iteration {iter:d} for rank {root:d}"),
            )

    @requires_gloo()
    def test_set_gloo_pg_timeout(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        pg.allreduce(torch.rand(10))
        self.assertEqual(pg.options._timeout, timedelta(seconds=50))
        pg._set_default_timeout(timedelta(seconds=23))
        self.assertEqual(pg.options._timeout, timedelta(seconds=23))

    @requires_gloo()
    def test_scatter_stress(self):
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone())

    @skip_but_pass_in_sandcastle(
        "Test is flaky, see https://github.com/pytorch/pytorch/issues/15963"
    )
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_scatter_stress_cuda(self):
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()
    def test_gather_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element input tensor list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element input tensor list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [t1, t1], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element output list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element output list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * self.world_size, [t1] * self.world_size], [t1], opts)

        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = "Incorrect output list size {}. Output list size should be {}"
        with self.assertRaisesRegex(
            RuntimeError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)

        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(
            RuntimeError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t2] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t3] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(RuntimeError, "requires empty output on non-root"):
            opts = c10d.GatherOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.gather([[t1] * self.world_size], [t1], opts)

    def _test_gather_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Preallocate tensors for input/output
        input = [fn(torch.tensor([self.rank]))]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the gather root and accumulate work items
        futures = []
        for i in range(self.world_size):
            opts = c10d.GatherOptions()
            opts.rootRank = i
            if i == self.rank:
                futures.append(pg.gather([outputs], input, opts).get_future())
            else:
                futures.append(pg.gather([], input, opts).get_future())

        # Wait for work to complete
        expected = [fn(torch.tensor([rank])) for rank in range(self.world_size)]
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            if i == self.rank:
                self.assertEqual(expected, result)

    @requires_gloo()
    def test_gather_basics(self):
        self._test_gather_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_gather_basics_cuda(self):
        self._test_gather_basics(lambda t: t.clone().cuda())

    @requires_gloo()
    def test_gather_noncontiguous_input(self):
        # Take a column of 2D tensor, such that memory is not dense
        self._test_gather_basics(lambda t: t.expand(2, 2).tril().contiguous()[:, 0])

    def _test_gather_stress(self, inputs, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = []
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.GatherOptions()
                opts.rootRank = root
                if root == self.rank:
                    fut = pg.gather(outputs[i], [fn(inputs[i])], opts).get_future()
                else:
                    fut = pg.gather([], [fn(inputs[i])], opts).get_future()
                future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                result = future_handle.value()
                self.assertEqual(
                    expected_outputs[iter],
                    [result],
                    msg=(f"Mismatch in iteration {iter:d} for root {root:d}"),
                )

    @requires_gloo()
    def test_gather_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    @requires_gloo()
    def test_gather_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()
    def test_allgather_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(
            RuntimeError, "requires non-empty input tensor list"
        ):
            pg.allgather([], [])

        with self.assertRaisesRegex(
            RuntimeError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([], [t1])

        with self.assertRaisesRegex(
            RuntimeError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([[t1] * self.world_size, [t1] * self.world_size], [t1])

        with self.assertRaisesRegex(RuntimeError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size - 1)], [t1])

        with self.assertRaisesRegex(RuntimeError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size + 1)], [t1])

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t2]
            )

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t3]
            )

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            pg.allgather([([t1, t2] * (self.world_size))[: self.world_size]], [t1])

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            pg.allgather([([t1, t3] * (self.world_size))[: self.world_size]], [t1])

    def _test_allgather_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Run with N input tensor per rank
        for n in [1, 2, 3]:
            input = [fn(torch.tensor([n * self.rank + i])) for i in range(n)]
            output = [
                [fn(torch.tensor([-1])) for _ in range(n * self.world_size)]
                for _ in range(n)
            ]
            expected_output = [
                [fn(torch.tensor([i])) for i in range(n * self.world_size)]
                for _ in range(n)
            ]
            fut = pg.allgather(output, input).get_future()
            fut.wait()
            result = fut.value()
            if n == 1:
                result = [result]
            self.assertEqual(expected_output, result)

    @requires_gloo()
    def test_allgather_basics(self):
        self._test_allgather_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allgather_basics_cuda(self):
        self._test_allgather_basics(lambda t: t.clone().cuda())

    @requires_gloo()
    def test_allgather_noncontiguous_input(self):
        # Take a column of 2D tensor, such that memory is not dense
        self._test_allgather_basics(lambda t: t.expand(2, 2).tril().contiguous()[:, 0])

    @requires_gloo()
    def test_allgather_inference_mode(self):
        with torch.inference_mode():
            self._test_allgather_basics(lambda t: t.clone())

    def _test_allgather_stress(self, inputs, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = []
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        input_holder = {}
        for i in range(len(inputs)):
            # Note that this works around the data race discussed in
            # https://github.com/pytorch/pytorch/issues/75529, but we should
            # actually be able to pass the list directly into allgather when
            # that race is fixed.
            input_holder[i] = [fn(inputs[i])]
            fut = pg.allgather(outputs[i], input_holder[i]).get_future()
            future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            self.assertEqual(
                expected_outputs[i],
                [result],
                msg=(f"Mismatch in iteration {i:d}"),
            )

    @requires_gloo()
    def test_allgather_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_allgather_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()

```



## High-Level Overview


This Python file contains 18 class(es) and 182 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RendezvousTCPTest`, `RendezvousEnvTest`, `TimeoutTest`, `ProcessGroupGlooTest`, `DistributedDataParallelTest`, `GlobalLocalUnusedParamModule`, `FindUnusedParamModule`, `IgnoredOutput`, `IgnoredOutputWithUnusedParameters`, `MyModule`, `TestModel`, `ReducerModule`, `ReducerTest`, `ProcessGroupGlooLazyInitTest`, `ProcessGroupGlooFRTest`, `CommTest`, `GlooProcessGroupWithDispatchedCollectivesTests`, `LargeCommTest`

**Functions defined**: `simple_reduce_tests`, `simple_coalesced_reduce_tests`, `simple_multi_input_reduce_tests`, `test_tcp_init`, `test_logging_init`, `test_default_store_timeout_gloo`, `_create_process_group_gloo`, `setUp`, `opts`, `test_multi_device_constructor`, `test_empty_tensors`, `test_broadcast_checks`, `_test_broadcast_basics`, `broadcast`, `test_broadcast_basics`, `test_broadcast_basics_cuda`, `_test_broadcast_stress`, `test_broadcast_stress`, `test_broadcast_stress_cuda`, `test_allreduce_checks`

**Key imports**: copy, json, logging, math, operator, os, pickle, random, sys, tempfile


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `json`
- `logging`
- `math`
- `operator`
- `os`
- `pickle`
- `random`
- `sys`
- `tempfile`
- `time`
- `datetime`: timedelta
- `functools`: reduce
- `itertools`: groupby
- `torch`
- `torch.distributed as c10d`
- `test_c10d_common`
- `torch.distributed as dist`
- `torch.nn.functional as F`
- `torch.testing._internal.common_utils as common`
- `torch.nn.parallel`: DistributedDataParallel


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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/test_c10d_gloo.py
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

- **File Documentation**: `test_c10d_gloo.py_docs.md`
- **Keyword Index**: `test_c10d_gloo.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
