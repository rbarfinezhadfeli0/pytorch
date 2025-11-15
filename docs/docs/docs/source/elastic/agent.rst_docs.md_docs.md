# Documentation: `docs/docs/source/elastic/agent.rst_docs.md`

## File Metadata

- **Path**: `docs/docs/source/elastic/agent.rst_docs.md`
- **Size**: 5,003 bytes (4.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/elastic/agent.rst`

## File Metadata

- **Path**: `docs/source/elastic/agent.rst`
- **Size**: 2,791 bytes (2.73 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
Elastic Agent
==============

.. automodule:: torch.distributed.elastic.agent
.. currentmodule:: torch.distributed.elastic.agent

Server
--------

.. automodule:: torch.distributed.elastic.agent.server

Below is a diagram of an agent that manages a local group of workers.

.. image:: agent_diagram.jpg

Concepts
--------

This section describes the high-level classes and concepts that
are relevant to understanding the role of the ``agent`` in torchelastic.

.. currentmodule:: torch.distributed.elastic.agent.server

.. autoclass:: ElasticAgent
   :members:

.. autoclass:: WorkerSpec
   :members:

.. autoclass:: WorkerState
   :members:

.. autoclass:: Worker
   :members:

.. autoclass:: WorkerGroup
   :members:

Implementations
-------------------

Below are the agent implementations provided by torchelastic.

.. currentmodule:: torch.distributed.elastic.agent.server.local_elastic_agent
.. autoclass:: LocalElasticAgent


Extending the Agent
---------------------

To extend the agent you can implement ``ElasticAgent`` directly, however
we recommend you extend ``SimpleElasticAgent`` instead, which provides
most of the scaffolding and leaves you with a few specific abstract methods
to implement.

.. currentmodule:: torch.distributed.elastic.agent.server
.. autoclass:: SimpleElasticAgent
   :members:
   :private-members:

.. autoclass:: torch.distributed.elastic.agent.server.api.RunResult


Watchdog in the Agent
---------------------

A named pipe based watchdog can be enabled in ``LocalElasticAgent`` if an
environment variable ``TORCHELASTIC_ENABLE_FILE_TIMER`` with value 1 has
been defined in the ``LocalElasticAgent`` process.
Optionally, another environment variable ``TORCHELASTIC_TIMER_FILE``
can be set with a unique file name for the named pipe. If the environment
variable ``TORCHELASTIC_TIMER_FILE`` is not set, ``LocalElasticAgent``
will internally create a unique file name and set it to the environment
variable ``TORCHELASTIC_TIMER_FILE``, and this environment variable will
be propagated to the worker processes to allow them to connect to the same
named pipe that ``LocalElasticAgent`` uses.


Health Check Server
-------------------

A health check monitoring server can be enabled in ``LocalElasticAgent``
if an environment variable ``TORCHELASTIC_HEALTH_CHECK_PORT`` has been defined
in the ``LocalElasticAgent`` process.
Adding interface for health check server which can be extended by starting tcp/http
server on the specified port number.
Additionally, health check server will have callback to check watchdog is alive.

.. automodule:: torch.distributed.elastic.agent.server.health_check_server

.. autoclass:: HealthCheckServer
   :members:

.. autofunction:: torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server

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

- No obvious security concerns detected in automated analysis.

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
- [`rendezvous.rst_docs.md`](./rendezvous.rst_docs.md)
- [`numa.rst_docs.md`](./numa.rst_docs.md)
- [`subprocess_handler.rst_docs.md`](./subprocess_handler.rst_docs.md)
- [`train_script.rst_docs.md`](./train_script.rst_docs.md)


## Cross-References

- **File Documentation**: `agent.rst_docs.md`
- **Keyword Index**: `agent.rst_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source/elastic`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source/elastic`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/docs/source/elastic`):

- [`subprocess_handler.rst_kw.md_docs.md`](./subprocess_handler.rst_kw.md_docs.md)
- [`multiprocessing.rst_kw.md_docs.md`](./multiprocessing.rst_kw.md_docs.md)
- [`customization.rst_docs.md_docs.md`](./customization.rst_docs.md_docs.md)
- [`kubernetes.rst_docs.md_docs.md`](./kubernetes.rst_docs.md_docs.md)
- [`metrics.rst_kw.md_docs.md`](./metrics.rst_kw.md_docs.md)
- [`control_plane.rst_docs.md_docs.md`](./control_plane.rst_docs.md_docs.md)
- [`run.rst_docs.md_docs.md`](./run.rst_docs.md_docs.md)
- [`events.rst_docs.md_docs.md`](./events.rst_docs.md_docs.md)
- [`timer.rst_docs.md_docs.md`](./timer.rst_docs.md_docs.md)
- [`metrics.rst_docs.md_docs.md`](./metrics.rst_docs.md_docs.md)


## Cross-References

- **File Documentation**: `agent.rst_docs.md_docs.md`
- **Keyword Index**: `agent.rst_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
