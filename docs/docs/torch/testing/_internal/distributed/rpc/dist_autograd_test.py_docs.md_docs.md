# Documentation: `docs/torch/testing/_internal/distributed/rpc/dist_autograd_test.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/distributed/rpc/dist_autograd_test.py_docs.md`
- **Size**: 54,220 bytes (52.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/distributed/rpc/dist_autograd_test.py`

## File Metadata

- **Path**: `torch/testing/_internal/distributed/rpc/dist_autograd_test.py`
- **Size**: 107,044 bytes (104.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
# mypy: allow-untyped-defs

import random
import sys
import threading
import time
from datetime import timedelta
from enum import Enum

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    IS_MACOS,
    skip_but_pass_in_sandcastle_if,
)
from torch.testing._internal.dist_utils import (
    dist_init,
    initialize_pg,
    wait_until_node_failure,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


# Right now we test up to 3-layer nested rpc calls.
# rpc_done[1] and ctx_ids[1] represent rpc is done in prev rank, and context id
# sent from prev rank respectively.
# rpc_done[2] and ctx_ids[2] represents for prev of prev rank.
# rpc_done[3] and ctx_ids[3] represents for prev of prev of prev rank.
# rpc_done[0] and ctx_ids[0] represents for current rank, but mostly not used.
rpc_done = [False, False, False, False]
ctx_ids = [-1, -1, -1, -1]

known_context_ids = set()

requires_grad_tensor = torch.ones(3, 3, requires_grad=True)


# Send rpc done info and context_id to
# dst_rank = (self.rank + rank_distance) % self.world_size
# we don't need a lock here since the GIL is held while executing remote
# python UDFs, so access is serialized across several workers.
def _set_rpc_done(ctx_id, rank_distance):
    global rpc_done
    global ctx_ids
    global known_context_ids
    rpc_done[rank_distance] = True
    ctx_ids[rank_distance] = ctx_id
    known_context_ids.add(ctx_id)


def _check_rpc_done(rank_distance):
    while not rpc_done[rank_distance]:
        time.sleep(0.1)


def _torch_ones(sizes, requires_grad=False):
    return torch.ones(sizes, requires_grad=requires_grad)


# This method must be called on the rref owner, and verifies that the grad of
# rref tensor equals to the given grad.
def _compare_owner_value(context_id, rref, grad):
    grads = dist_autograd.get_gradients(context_id)
    x = grads[rref.local_value()]
    if x.is_sparse:
        assert grad.is_sparse
        x = x.to_dense()
        grad = grad.to_dense()
    else:
        assert not grad.is_sparse
    return torch.equal(x, grad)


def create_tensor():
    return torch.ones((3, 3), requires_grad=True)


def build_sparse_tensor(coalesce=False, requires_grad=True, dtype=torch.float32):
    i = [[0, 1, 1], [2, 0, 2]]
    v = [3.2, 4.1, 5.3]
    tensor = torch.sparse_coo_tensor(
        i, v, (3, 3), requires_grad=requires_grad, dtype=dtype
    )
    if coalesce:
        tensor = tensor.coalesce()
    return tensor


@torch.jit.script
def create_torchscript_tensor() -> torch.Tensor:
    return torch.ones((3, 3)).requires_grad_()


def my_py_add(t1, t2):
    return torch.add(t1, t2)


def my_scalar_add(a, b):
    return a + b


def my_rref_add(rref_t1, t2):
    ret = torch.add(rref_t1.local_value(), t2)
    return ret


@torch.jit.script
def my_script_add(t1, t2):
    return torch.add(t1, t2)


@torch.jit.script
def my_script_ref_add(ref_t1: RRef[torch.Tensor], t2: torch.Tensor) -> torch.Tensor:
    t1 = ref_t1.to_here()
    return torch.add(t1, t2)


def my_nested_rref_add(dst, rref_t1, t2):
    return rpc.rpc_sync(dst, my_rref_add, args=(rref_t1, t2))


def ret_requires_grad():
    return requires_grad_tensor


def my_py_nested_call(t1, t2, dst, world_size, hops):
    next_dst = (dst + 1) % world_size
    if hops > 0:
        return rpc.rpc_sync(
            worker_name(next_dst),
            my_py_nested_call,
            args=(t1, t2, next_dst, world_size, hops - 1),
        )
    else:
        return rpc.rpc_sync(worker_name(next_dst), my_py_add, args=(t1, t2))


# after dist autograd context is cleaned up, it should be cleaned up on other
# nodes. This helper allows timeout_seconds for those RPCs to be completed, and
# ensures that all the contexts have been cleaned up in that timeframe.any
def _all_contexts_cleaned_up(timeout_seconds=10):
    global known_context_ids
    start = time.time()
    context_id_to_raised = set()
    while (
        time.time() - start < timeout_seconds
        and context_id_to_raised != known_context_ids
    ):
        for context_id in known_context_ids:
            try:
                dist_autograd._retrieve_context(context_id)
            except RuntimeError:
                context_id_to_raised.add(context_id)
    # all contexts have been cleaned up if trying to retrieve any context resulted in a RuntimeError.
    success = context_id_to_raised == known_context_ids
    return success


# This function creates a dis autograd context, run rpc_sync on the given ps,
# and then blocks until the ps has verified the grads are correctly accumulated.
def _run_trainer(rref_t1, t2, ps, rank_diff, sparse):
    with dist_autograd.context() as context_id:
        ret = rpc.rpc_sync(ps, my_rref_add, args=(rref_t1, t2))
        if sparse:
            loss = torch.sparse.sum(ret)
        else:
            loss = ret.sum()
        dist_autograd.backward(context_id, [loss])
        # prevent deleting dist autograd context
        rpc.rpc_sync(ps, _set_rpc_done, args=(context_id, rank_diff))
        rpc.rpc_sync(ps, _check_rpc_done, args=(0,))


# This function is the same as _run_trainer, except rpc calls torchscript
# function "my_script_ref_add" instead of python function "my_rref_add"
def _run_trainer_torchscript(rref_t1, t2, ps, rank_diff, sparse):
    with dist_autograd.context() as context_id:
        ret = rpc.rpc_sync(ps, my_script_ref_add, args=(rref_t1, t2))
        if sparse:
            loss = torch.sparse.sum(ret)
        else:
            loss = ret.sum()
        dist_autograd.backward(context_id, [loss])
        # prevent deleting dist autograd context
        rpc.rpc_sync(ps, _set_rpc_done, args=(context_id, rank_diff))
        rpc.rpc_sync(ps, _check_rpc_done, args=(0,))


class SimulateBackwardError(Function):
    _simulate_error = True

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    @once_differentiable
    def backward(ctx, input):
        if SimulateBackwardError._simulate_error:
            raise Exception("Simulate error on backward pass")  # noqa: TRY002
        else:
            return input


class ExecMode(Enum):
    LOCAL = 1  # Run the operation locally.
    RPC_SYNC = 2  # Run the operation using rpc_sync
    REMOTE = 3  # Run the operation using remote.
    RPC_ASYNC = 4  # Run the operation using rpc_async


# Common utils for both CPU and CUDA test suites
class CommonDistAutogradTest(RpcAgentTestFixture):
    def _exec_func_with_dst(self, dst, exec_mode, method, *args):
        if ExecMode.LOCAL == exec_mode:
            if len(args) == 1 and isinstance(args[0], list):
                return method(*args[0])
            return method(*args)
        elif ExecMode.RPC_SYNC == exec_mode:
            return rpc.rpc_sync(worker_name(dst), method, args=(args))
        elif ExecMode.REMOTE == exec_mode:
            return rpc.remote(worker_name(dst), method, args=(args)).to_here()
        elif ExecMode.RPC_ASYNC == exec_mode:
            fut = rpc.rpc_async(worker_name(dst), method, args=(args))
            return fut.wait()
        else:
            raise ValueError(f"Unrecognized ExecMode {exec_mode}")

    def _exec_func(self, exec_mode, method, *args):
        return self._exec_func_with_dst(self._next_rank(), exec_mode, method, *args)

    def _next_rank(self):
        if hasattr(self, "dst_rank"):
            self.dst_rank = (self.dst_rank + 1) % self.world_size
            if self.dst_rank == self.rank:
                return self._next_rank()
        else:
            self.dst_rank = (self.rank + 1) % self.world_size
        return self.dst_rank

    def _check_rpc_done(self, rank_distance):
        _check_rpc_done(rank_distance)

    def _verify_backwards(self, exec_mode, tensors, context_id, local_grads, *args):
        if exec_mode == ExecMode.LOCAL:
            torch.autograd.backward(tensors)
            return [arg.grad for arg in args]
        else:
            self._verify_backwards_remote(tensors, context_id, local_grads, *args)

    def _verify_backwards_remote(self, tensors, context_id, local_grads, *args):
        dist_autograd.backward(context_id, tensors)

        # Verify grads were accumulated appropriately.
        grads = dist_autograd.get_gradients(context_id)
        nargs = len(args)
        ngrads = 0
        for i in range(nargs):
            if local_grads[i] is not None:
                self.assertIn(args[i], grads)
                self.assertEqual(local_grads[i], grads[args[i]])
                ngrads += 1
            else:
                self.assertNotIn(args[i], grads)

        self.assertEqual(ngrads, len(grads))

    def _test_graph(self, fn, exec_mode, sparse):
        dst_rank = (self.rank + 1) % self.world_size

        initialize_pg(self.file_init_method, self.rank, self.world_size)

        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor()
                t2 = build_sparse_tensor()
            else:
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), fn, args=(t1, t2))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), fn, args=(t1, t2)).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))

            # Verify graph for current context id.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(
                next(iter(send_functions.values())),
                next(iter(recv_functions.values())),
                t1,
                t2,
                ret,
            )

            # Wait for the prev rank to be done with rpc.
            self._check_rpc_done(1)
            # Verify graph for previous context id.
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(next(iter(send_functions.values())))
            # this barrier is needed so one worker does not clean up their
            # autograd context before another worker tries to access it.
            dist.barrier()

        # autograd context should be cleaned up by now.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._retrieve_context(context_id)

        # No autograd context available.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._current_context()

    # 3-layer nested calls
    def _test_graph_for_py_nested_call(self, exec_mode, sparse):
        dst_rank = (self.rank + 1) % self.world_size

        initialize_pg(self.file_init_method, self.rank, self.world_size)

        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor(requires_grad=True)
                t2 = build_sparse_tensor(requires_grad=True)
            else:
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(
                    worker_name(dst_rank),
                    my_py_nested_call,
                    args=(t1, t2, dst_rank, self.world_size, 1),
                )
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(
                    worker_name(dst_rank),
                    my_py_nested_call,
                    args=(t1, t2, dst_rank, self.world_size, 1),
                ).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            # Barrier to ensure all RPCs are done.
            dist.barrier()

            for rd in [1, 2, 3]:
                rpc.rpc_sync(
                    worker_name((self.rank + rd) % self.world_size),
                    _set_rpc_done,
                    args=(context_id, rd),
                )

            # Barrier to ensure all set_rpc_done have completed.
            dist.barrier()

            # For self.rank, it has 4 graphs to verify
            # One is for current context id when this rank send first rpc call.
            # Second one is for prev context id when this rank make 1st nested
            # call.
            # Third one is for prev prev context id when this rank make
            # 2nd nested call.
            # Last one is for prev prev prev context id when this rank
            # execute the torch.add() operator.

            # Verify first graph for current context id.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(
                next(iter(send_functions.values())),
                next(iter(recv_functions.values())),
                t1,
                t2,
                ret,
            )

            # Verify second graph for 1st nested call.
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)

            # Verify third graph for 2nd nested call.
            ctx = dist_autograd._retrieve_context(ctx_ids[2])
            self._verify_graph_for_nested_rpc_call(ctx)

            # verify last graph for rpc call execution.
            ctx = dist_autograd._retrieve_context(ctx_ids[3])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(next(iter(send_functions.values())))
            # this barrier is needed so one worker does not clean up their
            # autograd context before another worker tries to access it.
            dist.barrier()

    # Rank0->Rank1->Rank0
    def _test_graph_for_py_nested_call_itself(self, exec_mode, sparse):
        dst_rank = (self.rank + 1) % self.world_size

        initialize_pg(self.file_init_method, self.rank, self.world_size)

        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor(requires_grad=True)
                t2 = build_sparse_tensor(requires_grad=True)
            else:
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(
                    worker_name(dst_rank),
                    my_py_nested_call,
                    args=(
                        t1,
                        t2,
                        (self.rank - 1 + self.world_size) % self.world_size,
                        self.world_size,
                        0,
                    ),
                )
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(
                    worker_name(dst_rank),
                    my_py_nested_call,
                    args=(
                        t1,
                        t2,
                        (self.rank - 1 + self.world_size) % self.world_size,
                        self.world_size,
                        0,
                    ),
                ).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            rpc.rpc_sync(
                worker_name((self.rank + 1) % self.world_size),
                _set_rpc_done,
                args=(context_id, 1),
            )

            # For self.rank, it has 2 graphs to verify.
            # One is for current context id when this rank send first rpc
            # call and execute the torch.add() operator.
            # Another one is for prev context id when this rank make
            # nested call.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(2, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(2, len(recv_functions))
            self._verify_graph_for_first_rpc_call(
                next(iter(send_functions.values())),
                list(recv_functions.values())[1],
                t1,
                t2,
                ret,
            )
            self._verify_graph_for_rpc_call_exec(list(send_functions.values())[1])

            # Verify two pairs of send and recv functions for nested
            # call
            self._check_rpc_done(1)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)
            # this barrier is needed so one worker does not clean up their
            # autograd context before another worker tries to access it.
            dist.barrier()

    def _test_no_graph_with_tensors_not_require_grad(self, exec_mode, sparse):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor(requires_grad=False)
                t2 = build_sparse_tensor(requires_grad=False)
            else:
                t1 = torch.ones(3, 3, requires_grad=False)
                t2 = torch.zeros(3, 3, requires_grad=False)
            if ExecMode.RPC_SYNC == exec_mode:
                rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
            elif ExecMode.REMOTE == exec_mode:
                rpc.remote(worker_name(dst_rank), torch.add, args=(t1, t2)).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))

            ctx = dist_autograd._current_context()
            send_functions = ctx._send_functions()
            self.assertEqual(len(send_functions), 0)
            recv_functions = ctx._recv_functions()
            self.assertEqual(len(recv_functions), 0)

            # Wait for the prev rank to be done with rpc.
            self._check_rpc_done(1)
            # NB: RRef.to_here() always passes the autograd context to the
            # the callee, as the caller does not know whether the return
            # value would contain a requires_grad tensor or not.
            #
            # rpc/remote with udf (_set_rpc_done here) also always passes the
            # autograd context to the callee due to the same reason.
            self.assertNotEqual(-1, dist_autograd._retrieve_context(ctx_ids[1]))
            dist.barrier()

    def _test_rpc_complex_args(self, exec_mode, sparse):
        with dist_autograd.context():
            num_tensors = 10
            tensors = []
            for i in range(num_tensors):
                if sparse:
                    tensor = build_sparse_tensor(requires_grad=(i % 2 == 0))
                else:
                    tensor = torch.ones(3, 3, requires_grad=(i % 2 == 0))
                tensors.append(tensor)
            dst_rank = self._next_rank()
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), torch.stack, args=(tensors,))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(
                    worker_name(dst_rank), torch.stack, args=(tensors,)
                ).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            self.assertEqual(torch.stack(tensors), ret)

            # Verify appropriate tensors have been attached the autograd graph.
            next_funcs = next(
                iter(dist_autograd._current_context()._send_functions().values())
            ).next_functions
            for i in range(len(next_funcs)):
                self.assertEqual(
                    "torch::autograd::AccumulateGrad", next_funcs[i][0].name()
                )
                self.assertEqual(tensors[i], next_funcs[i][0].variable)

            # Verify that the worker id has been recorded in the context
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(len(worker_ids), 1)
            self.assertEqual(worker_ids, {dst_rank})

    def context_cleanup_test_helper(self, rpc_args, func, nested=False):
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # test that in dist autograd, in the case that tensors communicated over RPC do
        # NOT require grad, we still cleanup the dist autograd contexts created
        # on other nodes. This is because the autograd context is still
        # communicated over RPC even if tensor arguments do not require grad, as
        #  it is possible that the response could.
        if nested:
            dst_rank = (self.rank + 1) % self.world_size
            nested_dst_rank = (dst_rank + 1) % self.world_size
            dst_ranks = {dst_rank}
        else:
            dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}

        with dist_autograd.context() as context_id:
            for dst_rank in dst_ranks:
                rpc.rpc_sync(worker_name(dst_rank), func, args=rpc_args)
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
                if nested:
                    rpc.rpc_sync(
                        worker_name(nested_dst_rank),
                        _set_rpc_done,
                        args=(context_id, 2),
                    )
        # the thread's context id should be cleaned up
        with self.assertRaises(RuntimeError):
            dist_autograd._retrieve_context(context_id)
        # Ensure all peers have finished mutating the
        # `known_context_ids` set.
        dist.barrier()
        # check that all contexts have been cleaned up.
        success = _all_contexts_cleaned_up()
        self.assertTrue(success)

    def _backward_no_grad_on_tensor(self, t1, t2, sparse):
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.add, args=(t1, t2)
            )
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            self.assertIsNone(t1.grad)
            self.assertIsNone(t2.grad)

            # Now populate .grad with local autograd engine and
            # verify dist autograd doesn't mess with it.
            loss_local = torch.add(t1, t2)
            if sparse:
                loss_local = torch.sparse.sum(loss_local)
            else:
                loss_local = loss_local.sum()
            loss_local.backward()
            self.assertIsNotNone(t1.grad)
            self.assertIsNotNone(t2.grad)

            t1_grad_before = t1.grad
            t2_grad_before = t2.grad
            dist_autograd.backward(context_id, [loss])
            self.assertEqual(t1_grad_before, t1.grad)
            self.assertEqual(t2_grad_before, t2.grad)

    # The current rank first creates a tensor on the rref_owner, and then passes
    # the rref with another tensor to the callee to run either my_rref_add or
    # my_nested_rref_add, depending on whether the callee is the rref owner.
    # The grad of tensor lives on the current rank, and the grad of the rref
    # tensor lives on the rref owner.
    def _backward_rref(self, callee, rref_owner, t1, t2, local_grads, sparse):
        local_ret = torch.add(t1, t2)
        if sparse:
            local_ret = torch.sparse.sum(local_ret)
        else:
            local_ret = local_ret.sum()
        local_ret.backward()
        with dist_autograd.context() as context_id:
            if sparse:
                rref_t1 = rpc.remote(
                    rref_owner,
                    build_sparse_tensor,
                    args=(
                        False,
                        True,
                    ),
                )
            else:
                rref_t1 = rpc.remote(
                    rref_owner,
                    _torch_ones,
                    args=((3, 3),),
                    kwargs={"requires_grad": True},
                )
            if callee == rref_owner:
                rref = rpc.remote(callee, my_rref_add, args=(rref_t1, t2))
            else:
                rref = rpc.remote(
                    callee, my_nested_rref_add, args=(rref_owner, rref_t1, t2)
                )
            ret = rref.to_here()
            if sparse:
                ret = torch.sparse.sum(ret)
            else:
                ret = ret.sum()
            dist_autograd.backward(context_id, [ret])

            # verify grads on caller
            grads = dist_autograd.get_gradients(context_id)
            self.assertIn(t2, grads)
            self.assertEqual(grads[t2], t2.grad)

            # verify grads on rref owner
            self.assertTrue(
                rpc.rpc_sync(
                    rref_owner,
                    _compare_owner_value,
                    args=(context_id, rref_t1, t1.grad),
                )
            )

    # In this test, every rank will serve as a parameter server (ps) and a
    # driver, and then kicks off trainers on the other three ranks. So, we have:
    # ps = rank0 with trainers = rank1/2/3
    # ps = rank2 with trainers = rank2/3/0
    # ps = rank3 with trainers = rank3/0/1
    # ps = rank4 with trainers = rank0/1/2
    #
    # These four test ps-trainer groups run on completely separate autograd
    # graphs, but they share the same set of underlying RpcAgents.
    def _test_trainer_ps(self, create_ref_fn, trainer_fn, sparse):
        if sparse:
            t1 = build_sparse_tensor(requires_grad=True)
            t2 = build_sparse_tensor(requires_grad=True)
        else:
            t1 = torch.ones((3, 3), requires_grad=True)
            t2 = torch.zeros((3, 3), requires_grad=True)

        local_ret = torch.add(t1, t2)
        if sparse:
            torch.sparse.sum(local_ret).backward()
        else:
            local_ret.sum().backward()

        # create rref on self
        rref_t1 = rpc.remote(worker_name(self.rank), create_ref_fn, args=())

        # kick off forward and backward pass on three other workers (trainers)
        rank_diffs = [1, 2, 3]
        futures = [
            rpc.rpc_async(
                worker_name((self.rank + rank_diff) % self.world_size),
                trainer_fn,
                args=(rref_t1, t2, worker_name(self.rank), rank_diff, sparse),
            )
            for rank_diff in rank_diffs
        ]

        # check if the trainers have done with their backward pass
        for rank_diff in rank_diffs:
            self._check_rpc_done(rank_diff)

        # trainers are done and holding the context for verification
        for rank_diff in rank_diffs:
            # make sure grads are accumulated for the same tensors and values
            # are all correct
            ctx_id = ctx_ids[rank_diff]
            grads = dist_autograd.get_gradients(ctx_id)
            local_t1 = rref_t1.to_here()
            self.assertIn(local_t1, grads)
            self.assertEqual(grads[local_t1], t1.grad)

        # unblock trainers
        _set_rpc_done(None, 0)

        # wait until all trainers are done
        torch.futures.wait_all(futures)

    def _backward_multiple_round_trips(self, t1, t2, t3, t4, t5, local_grads, sparse):
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                # Multiple RPCs between different nodes.
                val = self._exec_func(exec_mode, torch.add, t1, t2)
                val = self._exec_func(exec_mode, torch.mul, t3, val)
                s1 = self._exec_func(exec_mode, torch.stack, (t4, val))
                s2 = self._exec_func(exec_mode, torch.stack, (t5, val))
                if sparse:
                    val = self._exec_func(exec_mode, torch.mul, s1, s2)
                    val = self._exec_func(exec_mode, torch.mul, val, val)
                    loss = torch.sparse.sum(val)
                else:
                    val = self._exec_func(exec_mode, torch.bmm, s1, s2)
                    val = self._exec_func(exec_mode, torch.matmul, val, val)
                    loss = val.sum()

                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2, t3, t4, t5
                )
                local_grads = ret if ret else local_grads

    def _backward_different_dtypes(self, t1, t2, sparse):
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                loss = self._exec_func(exec_mode, torch.add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(loss)
                else:
                    loss = loss.sum()
                local_grads = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )

    # Run the same code locally and with dist autograd and verify gradients
    # are same.
    def _backward_simple_python_udf(self, t1, t2, sparse):
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, my_py_add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                local_grads = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )

    # Run the same code locally and with dist autograd and verify gradients
    # are same.
    def _backward_simple_script_call(self, t1, t2, sparse):
        local_grads = None
        for exec_mode in [
            ExecMode.LOCAL,
            ExecMode.RPC_SYNC,
            ExecMode.RPC_ASYNC,
            ExecMode.REMOTE,
        ]:
            with dist_autograd.context() as context_id:
                forward_ret = self._exec_func(exec_mode, my_script_add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(forward_ret)
                else:
                    loss = forward_ret.sum()
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )
                local_grads = ret if ret else local_grads

    def _nested_backward_accumulate_grads(self, t1, t2, sparse):
        with dist_autograd.context() as context_id:
            ret = rpc.rpc_sync(
                worker_name(self._next_rank()),
                DistAutogradTest._test_nested_backward_accumulate_grads,
                args=(t1, t2, self._next_rank()),
            )
            if sparse:
                loss = torch.sparse.sum(ret)
            else:
                loss = ret.sum()
            # Run backward twice.
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            dist_autograd.backward(context_id, [loss])

    def _backwards_nested_python_udf(self, t1, t2, sparse):
        t3 = t1 * t2
        t4 = t1 + t2
        res = t3 + t4
        loss = t1 * t2 * t3 * t4 * res
        if sparse:
            loss = torch.sparse.sum(loss)
        else:
            loss = loss.sum()
        torch.autograd.backward([loss])

        # Now run distributed autograd.
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()),
                DistAutogradTest._nested_python_udf,
                args=(t1, t2, self._next_rank()),
            )
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            dist_autograd.backward(context_id, [loss])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])

    def _mixed_requires_grad(self, t1, t2, sparse):
        for exec_mode in [ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(
                    exec_mode, DistAutogradTest._mixed_requires_grad_operaton, t1, t2
                )
                self.assertEqual(t1 * t2, ret)
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                dist_autograd.backward(context_id, [loss])
                self.assertTrue(t1.requires_grad)
                self.assertFalse(t2.requires_grad)
                grads = dist_autograd.get_gradients(context_id)
                self.assertIn(t1, grads)
                self.assertNotIn(t2, grads)
                self.assertEqual(t2, grads[t1])

    def _multiple_backward(self, t1, t2, sparse):
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.add, args=(t1, t2)
            )
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            # Run backward in a loop multiple times.
            for _ in range(1000):
                dist_autograd.backward(context_id, [loss], retain_graph=True)

    # For current context, this rank sends t1 and t2 tensors to dst_rank,
    # then get t3 = torch.add(t1, t2) result tensor.
    # For the current context in this rank, it expects graph like this:
    #  send function:
    #              rpcSendBackward
    #                  /          \
    #  t1.AccumulateGrad         t2.AccumulateGrad
    #
    #  recv function:
    #
    #            |
    #          t3.rpcRecvBackward
    #
    def _verify_graph_for_first_rpc_call(
        self, send_function, recv_function, t1, t2, ret
    ):
        # Retrieve the next functions in the graph.
        next_funcs = send_function.next_functions
        self.assertEqual(2, len(next_funcs))

        # We should now hit t1 and t2 in the autograd graph.
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[0][0].name())
        self.assertEqual(t1, next_funcs[0][0].variable)
        self.assertEqual(0, next_funcs[0][1])
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[1][0].name())
        self.assertEqual(t2, next_funcs[1][0].variable)
        self.assertEqual(0, next_funcs[1][1])

        # Test recv functions.
        self.assertEqual(ret.grad_fn, recv_function)

    # Run the same code locally and with dist autograd and verify gradients
    # are same.
    def _backward_simple(self, dst, t1, t2, local_grads, sparse):
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func_with_dst(dst, exec_mode, torch.add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )
                local_grads = ret if ret else local_grads

    # For a context passed from previous nested chain calls, this rank
    # receives two tensors t1 and t2, executes torch.add(t1, t2) and sends
    # result tensor t3 back.
    # For this context in this rank, it expects graph like this:
    #  send and recv functions:
    #       rpcSendBackward
    #           |
    #          t3.AddBackward0
    #          /             \
    # t1.recvRpcBackward    t2.recvRpcBackward
    def _verify_graph_for_rpc_call_exec(self, send_function):
        # Verify next function is AddBackward0
        next_funcs = send_function.next_functions
        self.assertEqual(1, len(next_funcs))
        add_backward_fn = next_funcs[0][0]
        self.assertEqual("AddBackward0", add_backward_fn.name())

        # Verify the next two functions are the same recv backward function.
        next_funcs = add_backward_fn.next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[1][0].name()
        )
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])

    # For a context passed from previous nested chain calls, this rank
    # receives two tensors t1 and t2, forwards t1 and t2 tensors using
    # nested rpc call to next dst. In return route, receive result tensor t3
    # from next dst and forwarding t3 back to previous calls.
    # For this context in this rank, it expects graph like this:
    #  send and recv functions for receiving and forwarding t1 and t2:
    #       rpcSendBackward
    #          /          \
    # t1.recvRpcBackward    t2.recvRpcBackward
    #  send and recv functions for receiving and forwarding t3:
    #       rpcSendBackward
    #             |
    #           t3.recvRpcBackward
    def _verify_graph_for_nested_rpc_call(self, ctx):
        send_functions = ctx._send_functions()
        self.assertEqual(2, len(send_functions))

        # For send function when making nest rpc call,
        # next functions of the send function are two recv functions
        # for received two tensors from previous call
        next_funcs = next(iter(send_functions.values())).next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[1][0].name()
        )
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])

        # For send function when returning response to previous call
        # next function of the send function is the recv function
        # for received tensor result returned from nested call
        next_funcs = list(send_functions.values())[1].next_functions
        self.assertEqual(1, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )


class TensorPipeAgentDistAutogradTest(CommonDistAutogradTest):
    # Sparse tests only work with TensorPipeAgent.
    @dist_init
    def test_graph_for_builtin_call_sparse(self):
        self._test_graph(torch.add, ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_python_call_sparse(self):
        self._test_graph(my_py_add, ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_builtin_remote_call_sparse(self):
        self._test_graph(torch.add, ExecMode.REMOTE, True)

    @dist_init
    def test_graph_for_python_remote_call_sparse(self):
        self._test_graph(my_py_add, ExecMode.REMOTE, True)

    @dist_init
    def test_graph_for_py_nested_call_sparse(self):
        self._test_graph_for_py_nested_call(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_py_nested_remote_call_sparse(self):
        self._test_graph_for_py_nested_call(ExecMode.REMOTE, True)

    @dist_init
    def test_graph_for_py_nested_call_itself_sparse(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_py_nested_remote_call_itself_sparse(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.REMOTE, True)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad_sparse(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad_remote_sparse(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.REMOTE, True)

    @dist_init
    def test_rpc_complex_args_sparse(self):
        self._test_rpc_complex_args(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_remote_complex_args_sparse(self):
        self._test_rpc_complex_args(ExecMode.REMOTE, True)

    @dist_init
    def test_context_cleanup_tensor_with_grad_sparse(self):
        t1 = build_sparse_tensor(requires_grad=True)
        t2 = build_sparse_tensor(requires_grad=True)
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)

    @dist_init
    def test_context_cleanup_tensor_no_grad_sparse(self):
        t1 = build_sparse_tensor(requires_grad=False)
        self.context_cleanup_test_helper(rpc_args=(t1, t1), func=torch.add)

    @dist_init
    def test_context_cleanup_nested_rpc_sparse(self):
        t1 = build_sparse_tensor(requires_grad=True)
        t2 = build_sparse_tensor(requires_grad=True)
        dst_rank = (self.rank + 1) % self.world_size
        args = (t1, t2, dst_rank, self.world_size, 0)
        self.context_cleanup_test_helper(
            rpc_args=args, func=my_py_nested_call, nested=True
        )

    @dist_init
    def test_backward_no_grad_on_tensor_sparse(self):
        self._backward_no_grad_on_tensor(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_backward_simple_sparse(self):
        self._backward_simple(
            self._next_rank(),
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_backward_simple_self_sparse(self):
        self._backward_simple(
            self.rank,
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_backward_rref_multi_sparse(self):
        if self.rank > 0:
            callee = "worker0"
            rref_owner = callee
            self._backward_rref(
                callee,
                rref_owner,
                build_sparse_tensor(requires_grad=True),
                build_sparse_tensor(requires_grad=True),
                None,
                True,
            )

    @dist_init
    def test_backward_rref_sparse(self):
        callee = worker_name(self._next_rank())
        rref_owner = callee
        self._backward_rref(
            callee,
            rref_owner,
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_backward_rref_nested_sparse(self):
        callee = worker_name((self.rank + 1) % self.world_size)
        rref_owner = worker_name((self.rank + 2) % self.world_size)
        self._backward_rref(
            callee,
            rref_owner,
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_trainer_ps_sparse(self):
        self._test_trainer_ps(build_sparse_tensor, _run_trainer, True)

    @dist_init
    def test_backward_multiple_round_trips_sparse(self):
        self._backward_multiple_round_trips(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=False),
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=False),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_backward_different_dtypes_sparse(self):
        self._backward_different_dtypes(
            build_sparse_tensor(requires_grad=True, dtype=torch.float32),
            build_sparse_tensor(requires_grad=True, dtype=torch.float64),
            True,
        )

    @dist_init
    def test_backward_simple_python_udf_sparse(self):
        self._backward_simple_python_udf(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_backward_simple_script_call_sparse(self):
        self._backward_simple_script_call(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_nested_backward_accumulate_grads_sparse(self):
        self._nested_backward_accumulate_grads(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_backwards_nested_python_udf_sparse(self):
        # Run equivalent of _nested_python_udf locally.
        self._backwards_nested_python_udf(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_mixed_requires_grad_sparse(self):
        self._mixed_requires_grad(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=False),
            True,
        )

    @dist_init
    def test_multiple_backward_sparse(self):
        self._multiple_backward(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_embedding_bag_with_no_grad_tensors(self):
        dst = self._next_rank()
        remote_embedding = rpc.remote(
            worker_name(dst),
            torch.nn.EmbeddingBag,
            args=(16, 16),
            kwargs={"mode": "sum", "sparse": True},
        )
        local_embedding = torch.nn.EmbeddingBag(16, 16, mode="sum", sparse=True)

        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
        # requires_grad = True to record send/recv functions
        per_sample_weights = torch.rand((8), requires_grad=True)
        offsets = torch.LongTensor([0, 4])

        local_res = local_embedding(input, offsets, per_sample_weights)

        # Run backward twice.
        torch.autograd.backward([local_res.sum()], retain_graph=True)
        torch.autograd.backward([local_res.sum()])
        local_grad = local_embedding.weight.grad

        with dist_autograd.context() as context_id:
            res = rpc.rpc_sync(
                worker_name(dst),
                DistAutogradTest._call_remote_embedding,
                args=(remote_embedding, input, offsets, per_sample_weights),
            )

            # Run backward twice to test accumulation of sparse gradients.
            dist_autograd.backward(context_id, [res.sum()], retain_graph=True)
            dist_autograd.backward(context_id, [res.sum()])

            remote_grad = rpc.rpc_sync(
                worker_name(dst),
                DistAutogradTest._get_grad,
                args=(remote_embedding, context_id),
            )

            self.assertEqual(local_grad, remote_grad)


class DistAutogradTest(CommonDistAutogradTest):
    @dist_init
    def test_autograd_context(self):
        # Verify max possible id.
        max_auto_increment = 281474976710655
        self.assertEqual(
            max_auto_increment + (self.worker_id << 48), dist_autograd._get_max_id()
        )

        context_ids = []
        for _ in range(200):
            with dist_autograd.context() as context_id:
                self.assertEqual(
                    context_id,
                    dist_autograd._retrieve_context(context_id)._context_id(),
                )
                # First 16 bits should be worker_id.
                self.assertEqual(self.worker_id, context_id >> 48)
                context_ids.append(context_id)

        for context_id in context_ids:
            with self.assertRaisesRegex(
                RuntimeError,
                f"Could not find autograd context with id: {context_id}",
            ):
                dist_autograd._retrieve_context(context_id)

    @dist_init
    def test_nested_context(self):
        with (
            dist_autograd.context(),
            self.assertRaisesRegex(
                RuntimeError, "Already have an autograd context id for this thread"
            ),
            dist_autograd.context(),
        ):
            pass

    @dist_init
    def test_graph_for_builtin_call(s
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal/distributed/rpc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/torch/testing/_internal/distributed/rpc/dist_autograd_test.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal/distributed/rpc`):

- [`rpc_test.py_kw.md_docs.md`](./rpc_test.py_kw.md_docs.md)
- [`dist_optimizer_test.py_docs.md_docs.md`](./dist_optimizer_test.py_docs.md_docs.md)
- [`rpc_test.py_docs.md_docs.md`](./rpc_test.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`rpc_agent_test_fixture.py_kw.md_docs.md`](./rpc_agent_test_fixture.py_kw.md_docs.md)
- [`dist_optimizer_test.py_kw.md_docs.md`](./dist_optimizer_test.py_kw.md_docs.md)
- [`tensorpipe_rpc_agent_test_fixture.py_kw.md_docs.md`](./tensorpipe_rpc_agent_test_fixture.py_kw.md_docs.md)
- [`faulty_agent_rpc_test.py_docs.md_docs.md`](./faulty_agent_rpc_test.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `dist_autograd_test.py_docs.md_docs.md`
- **Keyword Index**: `dist_autograd_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
