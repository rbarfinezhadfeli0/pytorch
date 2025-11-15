# Documentation: `docs/source/elastic/rendezvous.rst`

## File Metadata

- **Path**: `docs/source/elastic/rendezvous.rst`
- **Size**: 2,909 bytes (2.84 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
.. _rendezvous-api:

Rendezvous
==========

.. automodule:: torch.distributed.elastic.rendezvous

Below is a state diagram describing how rendezvous works.

.. image:: etcd_rdzv_diagram.png

Registry
--------

.. autoclass:: RendezvousParameters
   :members:

.. autoclass:: RendezvousHandlerRegistry

.. automodule:: torch.distributed.elastic.rendezvous.registry

Handler
-------

.. currentmodule:: torch.distributed.elastic.rendezvous

.. autoclass:: RendezvousHandler
   :members:

Dataclasses
-----------
.. autoclass:: RendezvousInfo

.. currentmodule:: torch.distributed.elastic.rendezvous.api

.. autoclass:: RendezvousStoreInfo

   .. automethod:: build(rank, store)

Exceptions
----------
.. autoclass:: RendezvousError
.. autoclass:: RendezvousClosedError
.. autoclass:: RendezvousTimeoutError
.. autoclass:: RendezvousConnectionError
.. autoclass:: RendezvousStateError
.. autoclass:: RendezvousGracefulExitError

Implementations
---------------

Dynamic Rendezvous
******************

.. currentmodule:: torch.distributed.elastic.rendezvous.dynamic_rendezvous

.. autofunction:: create_handler

.. autoclass:: DynamicRendezvousHandler()
   :members: from_backend

.. autoclass:: RendezvousBackend
   :members:

.. autoclass:: RendezvousTimeout
   :members:

C10d Backend
^^^^^^^^^^^^

.. currentmodule:: torch.distributed.elastic.rendezvous.c10d_rendezvous_backend

.. autofunction:: create_backend

.. autoclass:: C10dRendezvousBackend
   :members:

Etcd Backend
^^^^^^^^^^^^

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_rendezvous_backend

.. autofunction:: create_backend

.. autoclass:: EtcdRendezvousBackend
   :members:

Etcd Rendezvous (Legacy)
************************

.. warning::
    The ``DynamicRendezvousHandler`` class supersedes the ``EtcdRendezvousHandler``
    class, and is recommended for most users. ``EtcdRendezvousHandler`` is in
    maintenance mode and will be deprecated in the future.

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_rendezvous

.. autoclass:: EtcdRendezvousHandler

Etcd Store
**********

The ``EtcdStore`` is the C10d ``Store`` instance type returned by
``next_rendezvous()`` when etcd is used as the rendezvous backend.

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_store

.. autoclass:: EtcdStore
   :members:

Etcd Server
***********

The ``EtcdServer`` is a convenience class that makes it easy for you to
start and stop an etcd server on a subprocess. This is useful for testing
or single-node (multi-worker) deployments where manually setting up an
etcd server on the side is cumbersome.

.. warning:: For production and multi-node deployments please consider
             properly deploying a highly available etcd server as this is
             the single point of failure for your distributed jobs.

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_server

.. autoclass:: EtcdServer

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source/elastic`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source/elastic`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`docs/source/elastic`):

- [`examples.rst_docs.md`](./examples.rst_docs.md)
- [`events.rst_docs.md`](./events.rst_docs.md)
- [`run.rst_docs.md`](./run.rst_docs.md)
- [`metrics.rst_docs.md`](./metrics.rst_docs.md)
- [`timer.rst_docs.md`](./timer.rst_docs.md)
- [`customization.rst_docs.md`](./customization.rst_docs.md)
- [`numa.rst_docs.md`](./numa.rst_docs.md)
- [`subprocess_handler.rst_docs.md`](./subprocess_handler.rst_docs.md)
- [`train_script.rst_docs.md`](./train_script.rst_docs.md)


## Cross-References

- **File Documentation**: `rendezvous.rst_docs.md`
- **Keyword Index**: `rendezvous.rst_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
