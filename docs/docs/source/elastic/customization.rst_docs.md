# Documentation: `docs/source/elastic/customization.rst`

## File Metadata

- **Path**: `docs/source/elastic/customization.rst`
- **Size**: 3,584 bytes (3.50 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
Customization
=============

This section describes how to customize TorchElastic to fit your needs.

Launcher
------------------------

The launcher program that ships with TorchElastic
should be sufficient for most use-cases (see :ref:`launcher-api`).
You can implement a custom launcher by
programmatically creating an agent and passing it specs for your workers as
shown below.

.. code-block:: python

  # my_launcher.py

  if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    rdzv_handler = RendezvousHandler(...)
    spec = WorkerSpec(
        local_world_size=args.nproc_per_node,
        fn=trainer_entrypoint_fn,
        args=(trainer_entrypoint_fn args.fn_args,...),
        rdzv_handler=rdzv_handler,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
    )

    agent = LocalElasticAgent(spec, start_method="spawn")
    try:
        run_result = agent.run()
        if run_result.is_failed():
            print(f"worker 0 failed with: run_result.failures[0]")
        else:
            print(f"worker 0 return value is: run_result.return_values[0]")
    except Exception ex:
        # handle exception


Rendezvous Handler
------------------------

To implement your own rendezvous, extend ``torch.distributed.elastic.rendezvous.RendezvousHandler``
and implement its methods.

.. warning:: Rendezvous handlers are tricky to implement. Before you begin
          make sure you completely understand the properties of rendezvous.
          Please refer to :ref:`rendezvous-api` for more information.

Once implemented you can pass your custom rendezvous handler to the worker
spec when creating the agent.

.. code-block:: python

    spec = WorkerSpec(
        rdzv_handler=MyRendezvousHandler(params),
        ...
    )
    elastic_agent = LocalElasticAgent(spec, start_method=start_method)
    elastic_agent.run(spec.role)


Metric Handler
-----------------------------

TorchElastic emits platform level metrics (see :ref:`metrics-api`).
By default metrics are emitted to `/dev/null` so you will not see them.
To have the metrics pushed to a metric handling service in your infrastructure,
implement a `torch.distributed.elastic.metrics.MetricHandler` and `configure` it in your
custom launcher.

.. code-block:: python

  # my_launcher.py

  import torch.distributed.elastic.metrics as metrics

  class MyMetricHandler(metrics.MetricHandler):
      def emit(self, metric_data: metrics.MetricData):
          # push metric_data to your metric sink

  def main():
    metrics.configure(MyMetricHandler())

    spec = WorkerSpec(...)
    agent = LocalElasticAgent(spec)
    agent.run()

Events Handler
-----------------------------

TorchElastic supports events recording (see :ref:`events-api`).
The events module defines API that allows you to record events and
implement custom EventHandler. EventHandler is used for publishing events
produced during torchelastic execution to different sources, e.g.  AWS CloudWatch.
By default it uses `torch.distributed.elastic.events.NullEventHandler` that ignores
events. To configure custom events handler you need to implement
`torch.distributed.elastic.events.EventHandler` interface and `configure` it
in your custom launcher.

.. code-block:: python

  # my_launcher.py

  import torch.distributed.elastic.events as events

  class MyEventHandler(events.EventHandler):
      def record(self, event: events.Event):
          # process event

  def main():
    events.configure(MyEventHandler())

    spec = WorkerSpec(...)
    agent = LocalElasticAgent(spec)
    agent.run()

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

- **Error Handling**: Includes exception handling


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
- [`rendezvous.rst_docs.md`](./rendezvous.rst_docs.md)
- [`numa.rst_docs.md`](./numa.rst_docs.md)
- [`subprocess_handler.rst_docs.md`](./subprocess_handler.rst_docs.md)
- [`train_script.rst_docs.md`](./train_script.rst_docs.md)


## Cross-References

- **File Documentation**: `customization.rst_docs.md`
- **Keyword Index**: `customization.rst_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
