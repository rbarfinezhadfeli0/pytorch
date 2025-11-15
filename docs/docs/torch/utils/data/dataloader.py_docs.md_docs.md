# Documentation: `docs/torch/utils/data/dataloader.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/dataloader.py_docs.md`
- **Size**: 53,918 bytes (52.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/dataloader.py`

## File Metadata

- **Path**: `torch/utils/data/dataloader.py`
- **Size**: 80,958 bytes (79.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
r"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter.

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""

from __future__ import annotations

import functools
import itertools
import logging
import multiprocessing as python_multiprocessing
import os
import queue
import threading
import warnings
from collections.abc import Callable
from typing import Any, Generic, NoReturn, TYPE_CHECKING, TypeVar
from typing_extensions import Self

import torch
import torch.distributed as dist
import torch.utils.data.graph_settings
from torch._utils import ExceptionWrapper
from torch.utils.data import _utils
from torch.utils.data.datapipes.datapipe import (
    _IterDataPipeSerializationWrapper,
    _MapDataPipeSerializationWrapper,
    IterDataPipe,
    MapDataPipe,
)
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "DataLoader",
    "get_worker_info",
    "default_collate",
    "default_convert",
]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_worker_init_fn_t = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`, but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[list[_T]], Any]


# These functions used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate: _collate_fn_t = _utils.collate.default_collate
default_convert = _utils.collate.default_convert

get_worker_info = _utils.worker.get_worker_info

logger = logging.getLogger(__name__)


class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )
        else:
            return _utils.fetch._IterableDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )


class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.

    Used as sampler for :class:`~torch.utils.data.IterableDataset`.
    """

    def __iter__(self):
        while True:
            yield None


def _get_distributed_settings():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id) -> None:
    global_worker_id = worker_id
    info = torch.utils.data.get_worker_info()
    if info is None:
        raise AssertionError("Worker info is None in sharding worker init function")
    total_workers = info.num_workers
    datapipe = info.dataset
    if not isinstance(datapipe, (IterDataPipe, MapDataPipe)):
        raise AssertionError(
            "datapipe must be an instance of IterDataPipe or MapDataPipe"
        )
    # To distribute elements across distributed process evenly, we should shard data on distributed
    # processes first then shard on worker processes
    total_workers *= world_size
    global_worker_id = global_worker_id * world_size + rank_id
    # For BC, use default SHARDING_PRIORITIES
    torch.utils.data.graph_settings.apply_sharding(
        datapipe, total_workers, global_worker_id
    )
    if worker_init_fn is not None:
        worker_init_fn(worker_id)


def _share_dist_seed(generator, pg):
    _shared_seed = torch.empty((), dtype=torch.int64).random_(generator=generator)
    if isinstance(pg, dist.ProcessGroup):
        dist.broadcast(_shared_seed, src=0, group=pg)
    return _shared_seed.item()


class DataLoader(Generic[_T_co]):
    r"""
    Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (Callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (Callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        multiprocessing_context (str or multiprocessing.context.BaseContext, optional): If
            ``None``, the default
            `multiprocessing context <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_ # noqa: D401
            of your operating system will
            be used. (default: ``None``)
        generator (torch.Generator, optional): If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            ``base_seed`` for workers. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers batches prefetched across all workers. (default value depends
            on the set value for num_workers. If value of num_workers=0 default is ``None``.
            Otherwise, if value of ``num_workers > 0`` default is ``2``).
        persistent_workers (bool, optional): If ``True``, the data loader will not shut down
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
        pin_memory_device (str, optional): Deprecated, the current :ref:`accelerator<accelerators>`
            will be used as the device if ``pin_memory=True``.
        in_order (bool, optional): If ``False``, the data loader will not enforce that batches
            are returned in a first-in, first-out order. Only applies when ``num_workers > 0``. (default: ``True``)


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.

    .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                 :ref:`data-loading-randomness` notes for random seed related questions.

    .. warning:: Setting `in_order` to `False` can harm reproducibility and may lead to a skewed data
                 distribution being fed to the trainer in cases with imbalanced data.
    """

    dataset: Dataset[_T_co]
    batch_size: int | None
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Sampler | Iterable
    pin_memory_device: str
    prefetch_factor: int | None
    _iterator: _BaseDataLoaderIter | None
    __initialized = False

    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[list] | Iterable[list] | None = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: _worker_init_fn_t | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> None:
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; "
                "use num_workers=0 to disable multiprocessing."
            )

        if timeout < 0:
            raise ValueError("timeout option should be non-negative")

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing."
                "let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None."
            )
        elif num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError("prefetch_factor option should be non-negative")

        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs num_workers > 0")

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.in_order = in_order

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   _DataPipeSerializationWrapper container makes it easier to serialize without redefining pickler
        if isinstance(self.dataset, IterDataPipe):
            self.dataset = _IterDataPipeSerializationWrapper(self.dataset)
        elif isinstance(self.dataset, MapDataPipe):
            self.dataset = _MapDataPipeSerializationWrapper(self.dataset)

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and IterableDataset ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if isinstance(dataset, IterDataPipe):
                if shuffle is not None:
                    dataset = torch.utils.data.graph_settings.apply_shuffle_settings(
                        dataset, shuffle=shuffle
                    )
            # We cannot check `shuffle is not None` here, since previously `shuffle=False` was the default.
            elif shuffle not in {False, None}:
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle={shuffle}"
                )

            if sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified sampler option, but got sampler={sampler}"
                )
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    f"batch_sampler option, but got batch_sampler={batch_sampler}"
                )
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError(
                    "batch_size=None option disables auto-batching "
                    "and is mutually exclusive with drop_last"
                )

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                else:
                    sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = (
            None  # See NOTE [ IterableDataset and __len__ ]
        )

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital("Dataloader", "enabled", "True")  # type: ignore[attr-defined]

    def _get_iterator(self) -> _BaseDataLoaderIter:
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context) -> None:
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = torch.multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            "multiprocessing_context option "
                            f"should specify a valid start method in {valid_start_methods!r}, but got "
                            f"multiprocessing_context={multiprocessing_context!r}"
                        )
                    multiprocessing_context = torch.multiprocessing.get_context(
                        multiprocessing_context
                    )

                if not isinstance(
                    multiprocessing_context, python_multiprocessing.context.BaseContext
                ):
                    raise TypeError(
                        "multiprocessing_context option should be a valid context "
                        "object or a string specifying the start method, but got "
                        f"multiprocessing_context={multiprocessing_context}"
                    )
            else:
                raise ValueError(
                    "multiprocessing_context can only be used with "
                    "multi-process loading (num_workers > 0), but got "
                    f"num_workers={self.num_workers}"
                )

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val) -> None:
        if self.__initialized and attr in (
            "batch_size",
            "batch_sampler",
            "sampler",
            "drop_last",
            "dataset",
            "persistent_workers",
        ):
            raise ValueError(
                f"{attr} attribute should not be set after {self.__class__.__name__} is initialized"
            )

        super().__setattr__(attr, val)

    def __iter__(self) -> _BaseDataLoaderIter:
        # When using a single worker the returned iterator should be
        # created every time to avoid resetting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            # NOTE [ IterableDataset and __len__ ]
            #
            # For `IterableDataset`, `__len__` could be inaccurate when one naively
            # does multi-processing data loading, since the samples will be duplicated.
            # However, no real use case should be actually using that behavior, so
            # it should count as a user error. We should generally trust user
            # code to do the proper thing (e.g., configure each replica differently
            # in `__iter__`), and give us the correct `__len__` if they choose to
            # implement it (this will still throw if the dataset does not implement
            # a `__len__`).
            #
            # To provide a further warning, we track if `__len__` was called on the
            # `DataLoader`, save the returned value in `self._len_called`, and warn
            # if the iterator ends up yielding more than this number of samples.

            # Cannot statically verify that dataset is Sized
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore[assignment, arg-type]
            if (
                self.batch_size is not None
            ):  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil

                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)

    def check_worker_number_rationality(self) -> None:
        # This function check whether the dataloader's worker number is rational based on
        # current system's resource. Current rule is that if the number of workers this
        # Dataloader will create is bigger than the number of logical cpus that is allowed to
        # use, than we will pop up a warning to let user pay attention.
        #
        # eg. If current system has 2 physical CPUs with 16 cores each. And each core support 2
        #     threads, then the total logical cpus here is 2 * 16 * 2 = 64. Let's say current
        #     DataLoader process can use half of them which is 32, then the rational max number of
        #     worker that initiated from this process is 32.
        #     Now, let's say the created DataLoader has num_works = 40, which is bigger than 32.
        #     So the warning message is triggered to notify the user to lower the worker number if
        #     necessary.
        #
        #
        # [Note] Please note that this function respects `cpuset` only when os.sched_getaffinity is
        #        available (available in most of Linux system, but not OSX and Windows).
        #        When os.sched_getaffinity is not available, os.cpu_count() is called instead, but
        #        it doesn't respect cpuset.
        #        We don't take threading into account since each worker process is single threaded
        #        at this time.
        #
        #        We don't set any threading flags (eg. OMP_NUM_THREADS, MKL_NUM_THREADS, etc)
        #        other than `torch.set_num_threads` to 1 in the worker process, if the passing
        #        in functions use 3rd party modules that rely on those threading flags to determine
        #        how many thread to create (eg. numpy, etc), then it is caller's responsibility to
        #        set those flags correctly.
        def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):
            suggested_max_worker_msg = (
                (
                    (
                        "Our suggested max number of worker in current system is {}{}, which is smaller "
                        "than what this DataLoader is going to create."
                    ).format(
                        num_worker_suggest,
                        (
                            ""
                            if cpuset_checked
                            else " (`cpuset` is not taken into account)"
                        ),
                    )
                )
                if num_worker_suggest is not None
                else (
                    "DataLoader is not able to compute a suggested max number of worker in current system."
                )
            )

            warn_msg = (
                f"This DataLoader will create {num_worker_created} worker processes in total. {suggested_max_worker_msg} "
                "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
                "lower the worker number to avoid potential slowness/freeze if necessary."
            )
            return warn_msg

        if not self.num_workers or self.num_workers == 0:
            return

        # try to compute a suggested max number of worker based on system's resource
        max_num_worker_suggest = None
        cpuset_checked = False
        if hasattr(os, "sched_getaffinity"):
            try:
                max_num_worker_suggest = len(os.sched_getaffinity(0))
                cpuset_checked = True
            except Exception:
                pass
        if max_num_worker_suggest is None:
            # os.cpu_count() could return Optional[int]
            # get cpu count first and check None in order to satisfy mypy check
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_num_worker_suggest = cpu_count

        if max_num_worker_suggest is None:
            warnings.warn(
                _create_warning_msg(
                    max_num_worker_suggest, self.num_workers, cpuset_checked
                ),
                stacklevel=2,
            )
            return

        if self.num_workers > max_num_worker_suggest:
            warnings.warn(
                _create_warning_msg(
                    max_num_worker_suggest, self.num_workers, cpuset_checked
                ),
                stacklevel=2,
            )


class _BaseDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._shared_seed = None
        self._pg = None
        if isinstance(self._dataset, IterDataPipe):
            if dist.is_available() and dist.is_initialized():
                self._pg = dist.new_group(backend="gloo")
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(
                self._dataset, shared_rng
            )
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        ws, rank = _get_distributed_settings()
        self._world_size = ws
        self._rank = rank

        if loader.pin_memory and loader.pin_memory_device:
            warnings.warn(
                "pin_memory_device is deprecated, the current accelerator will be used as the device,"
                f"ignore pin_memory_device='{loader.pin_memory_device}'.",
                stacklevel=2,
            )
        if loader.pin_memory and not torch.accelerator.is_available():
            warn_msg = (
                "'pin_memory' argument is set as true but no accelerator is found, "
                "then device pinned memory won't be used."
            )
            warnings.warn(warn_msg, stacklevel=2)

        # Enabling pin_memory in _BaseDataLoaderIter to support identical
        # behavior in forked implementations using _BaseDataLoaderIter.
        self._pin_memory = loader.pin_memory and torch.accelerator.is_available()

        # Set pin memory device based on the current accelerator.
        self._pin_memory_device = (
            acc.type
            if self._pin_memory
            and (acc := torch.accelerator.current_accelerator()) is not None
            else None
        )

        # Currently, pin_memory would raise error on the MPS backend (see
        # https://github.com/pytorch/pytorch/issues/86060), so forcibly
        # disable pin_memory on MPS. Remove this restriction once pinned
        # memory allocation for MPS is fixed.
        if self._pin_memory_device == "mps":
            self._pin_memory = False
            warn_msg = (
                "'pin_memory' argument is set as true but not supported on MPS now, "
                "device pinned memory won't be used."
            )
            warnings.warn(warn_msg, stacklevel=2)

        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = (
            torch.empty((), dtype=torch.int64)
            .random_(generator=loader.generator)
            .item()
        )
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = f"enumerate(DataLoader)#{self.__class__.__name__}.__next__"

    def __iter__(self) -> Self:
        return self

    def _reset(self, loader, first_iter=False) -> None:
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        if isinstance(self._dataset, IterDataPipe):
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(
                self._dataset, shared_rng
            )

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self) -> NoReturn:
        raise NotImplementedError

    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                # TODO(https://github.com/pytorch/pytorch/issues/76750)
                self._reset()  # type: ignore[call-arg]
            data = self._next_data()
            self._num_yielded += 1
            if (
                self._dataset_kind == _DatasetKind.Iterable
                and self._IterableDataset_len_called is not None
                and self._num_yielded > self._IterableDataset_len_called
            ):
                warn_msg = (
                    f"Length of IterableDataset {self._dataset} was reported to be {self._IterableDataset_len_called}"
                    f"(when accessing len(dataloader)), but {self._num_yielded} samples have been fetched. "
                )
                if self._num_workers > 0:
                    warn_msg += (
                        "For multiprocessing data-loading, this could be caused by not properly configuring the "
                        "IterableDataset replica at each worker. Please see "
                        "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples."
                    )
                warnings.warn(warn_msg, stacklevel=2)
            return data

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader) -> None:
        super().__init__(loader)
        if self._timeout != 0:
            raise AssertionError("_SingleProcessDataLoaderIter requires timeout == 0")
        if self._num_workers != 0:
            raise AssertionError(
                "_SingleProcessDataLoaderIter requires num_workers == 0"
            )

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Taking care of distributed sharding
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            # For BC, use default SHARDING_PRIORITIES
            torch.utils.data.graph_settings.apply_sharding(
                self._dataset, self._world_size, self._rank
            )

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler."""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed (e.g., releasing GPU memory).
    #      Naturally, we implement the shutdown logic in `__del__` of
    #      DataLoaderIterator.
    #
    #      We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers and `pin_memory_thread` to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when the
    #      iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may already be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children.
    #
    #      Thus, we register an `atexit` hook that sets a global flag
    #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
    #      reverse order of registration, we are guaranteed that this flag is
    #      set before library resources we use are freed (which, at least in
    #      CPython, is done via an `atexit` handler defined in
    #      `multiprocessing/util.py`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/util.py#L320-L362
    #      registered when an object requiring this mechanism is first
    #      created, e.g., `mp.Queue`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/context.py#L100-L103
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/queues.py#L29
    #      )
    #
    #      So in `__del__`, we check if `_utils.python_exit_status` is set or
    #      `None` (freed), and perform no-op if so.
    #
    #      However, simply letting library clean-up codes run can also be bad,
    #      because such codes (i.e., `multiprocessing.util._exit_function()`)
    #      include join putting threads for `mp.Queue`, which can be blocking.
    #      Hence, the main process putting threads are called with
    #      `cancel_join_thread` at creation.  See later section
    #      [ 3b. A process won't hang when putting into a queue; ]
    #      for more details.
    #
    #      Here are two example cases where library clean-up codes can run
    #      before `__del__` is called:
    #
    #        1. If we hold onto a reference to the iterator, it more often
    #           than not tries to do `multiprocessing` library cleaning before
    #           clearing the alive referenced objects (https://github.com/pytorch/pytorch/issues/48666)
    #           and thus prevents our cleaning-up code to run first.
    #
    #        2. A similar issue araises when a `DataLoader` is used in a subprocess.
    #           When a process ends, it shuts the all its daemonic children
    #           down with a SIGTERM (instead of joining them without a timeout).
    #           Similarly for threads, but by a different mechanism. This fact,
    #           together with a few implementation details of multiprocessing, forces
    #           us to make workers daemonic. All of our problems arise when a
    #           DataLoader is used in a subprocess, and are caused by multiprocessing
    #           code which looks more or less like this:
    #
    #               try:
    #                   your_function_using_a_dataloader()
    #               finally:
    #                   multiprocessing.util._exit_function()
    #
    #           The joining/termination mentioned above happens inside
    #           `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #           throws, the stack trace stored in the exception will prevent the
    #           frame which uses `DataLoaderIter` to be freed. If the frame has any
    #           reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #           its  `__del__`, which starts the shutdown procedure, will not be
    #           called. That, in turn, means that workers aren't notified. Attempting
    #           to join in `_exit_function` will then result in a hang.
    #
    #           For context, `_exit_function` is also registered as an `atexit` call.
    #           So it is unclear to me (@ssnl) why this is needed in a finally block.
    #           The code dates back to 2008 and there is no comment on the original
    #           PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #           the finally block and the `atexit` registration) that explains this.
    #
    #
    #      Finally, another choice is to just shutdown workers with logic in 1
    #      above whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #           always corresponding to a `get()`), hanging on `get()` can still
    #           happen when data in queue is corrupted (e.g., due to
    #           `cancel_join_thread` or unexpected exit).
    #
    #           For child exit, we set a timeout whenever we try to get data
    #           from `data_queue`, and check the workers' status on each timeout
    #           and error.
    #           See `_DataLoaderiter._get_batch()` and
    #           `_DataLoaderiter._try_get_data()` for details.
    #
    #           Additionally, for child exit on non-Windows platforms, we also
    #           register a SIGCHLD handler (which is supported on Windows) on
    #           the main process, which checks if any of the workers fail in the
    #           (Python) handler. This is more efficient and faster in detecting
    #           worker failures, compared to only using the above mechanism.
    #           See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
    #
    #           For `.get()` calls where the sender(s) is not the workers, we
    #           guard them with timeouts, and check the status of the sender
    #           when timeout happens:
    #             + in the workers, the `_utils.worker.ManagerWatchdog` class
    #               checks the status of the main process.
    #             + if `pin_memory=True`, when getting from `pin_memory_thread`,
    #               check `pin_memory_thread` status periodically until `.get()`
    #               returns or see that `pin_memory_thread` died.
    #
    #        b. A process won't hang when putting into a queue;
    #
    #           We use `mp.Queue` which has a separate background thread to put
    #           objects from an unbounded buffer array. The background thread is
    #           daemonic and usually automatically joined when the process
    #           *exits*.
    #
    #           In case that the receiver has ended abruptly while
    #           reading from the pipe, the join will hang forever.  The usual
    #           solution for this in Python is calling  `q.cancel_join_thread`,
    #           which prevents automatically joining it when finalizing
    #           (exiting).
    #
    #           Nonetheless, `cancel_join_thread` must only be called when the
    #           queue is **not** going to be read from or write into by another
    #           process, because it may hold onto a lock or leave corrupted data
    #           in the queue, leading other readers/writers to hang.
    #
    #           Hence,
    #             + For worker processes, we only do so (for their output
    #               queues, i.e., `worker_result_queue`) before exiting.
    #             + For `pin_memory_thread`, its output queue `data_queue` is a
    #               `queue.Queue` that does blocking `put` if the queue is full.
    #               So there is no above problem, but as a result, in
    #               `_pin_memory_loop`, we do need to  wrap the `put` in a loop
    #               that breaks not only upon success, but also when the main
    #               process stops reading, i.e., is shutting down.
    #             + For loader process, we `cancel_join_thread()` for all
    #               `_index_queues` because the whole purpose of workers and
    #               `pin_memory_thread` is to serve the loader process.  If
    #               loader process is already exiting, we don't really care if
    #               the queues are corrupted.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # `workers_done_event`:
    #   A `multiprocessing.Event` shared among the main process and all worker
    #   processes. This is used to signal the workers that the iterator is
    #   shutting down. After it is set, they will not send processed data to
    #   queues anymore, and only wait for the final `None` before exiting.
    #   `done_event` isn't strictly needed. I.e., we can just check for `None`
    #   from the input queue, but it allows us to skip wasting resources
    #   processing data if we are already shutting down.
    #
    # `pin_memory_thread_done_event`:
    #   A `threading.Event` for a similar purpose to that of
    #   `workers_done_event`, but is for the `pin_memory_thread`. The reason
    #   that separate events are needed is that `pin_memory_thread` reads from
    #   the output queue of the workers. But the workers, upon seeing that
    #   `workers_done_event` is set, only wants to see the final `None`, and is
    #   not required to flush all data in the output queue (e.g., it may call
    #   `cancel_join_thread` on that queue if its `IterableDataset` iterator
    #   happens to exhaust coincidentally, which is out of the control of the
    #   main process). Thus, since we will exit `pin_memory_thread` before the
    #   workers (see below), two separate events are used.
    #
    # NOTE: In short, the protocol is that the main process will set these
    #       `done_event`s and then the corresponding processes/threads a `None`,
    #       and that they may exit at any time after receiving the `None`.
    #
    # NOTE: Using `None` as the final signal is valid, since normal data will
    #       always be a 2-tuple with the 1st element being the index of the data
    #       transferred (different from dataset index/key), and the 2nd being
    #       either the dataset key or the data sample (depending on which part
    #       of the data model the queue is at).
    #
    # [ worker processes ]
    #   Wh
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/data`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/data`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/utils/data`):

- [`backward_compatibility.py_kw.md_docs.md`](./backward_compatibility.py_kw.md_docs.md)
- [`graph_settings.py_kw.md_docs.md`](./graph_settings.py_kw.md_docs.md)
- [`graph.py_kw.md_docs.md`](./graph.py_kw.md_docs.md)
- [`sampler.py_kw.md_docs.md`](./sampler.py_kw.md_docs.md)
- [`dataloader.py_kw.md_docs.md`](./dataloader.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`dataset.py_kw.md_docs.md`](./dataset.py_kw.md_docs.md)
- [`distributed.py_docs.md_docs.md`](./distributed.py_docs.md_docs.md)
- [`sampler.py_docs.md_docs.md`](./sampler.py_docs.md_docs.md)
- [`dataset.py_docs.md_docs.md`](./dataset.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `dataloader.py_docs.md_docs.md`
- **Keyword Index**: `dataloader.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
