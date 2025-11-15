# Keyword Index: `torch/_inductor/remote_cache.py`

## File Information

- **Original File**: [torch/_inductor/remote_cache.py](../../../torch/_inductor/remote_cache.py)
- **Documentation**: [`remote_cache.py_docs.md`](./remote_cache.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`RedisRemoteCache`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RedisRemoteCacheBackend`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteAOTAutogradCache`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteAutotuneCache`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteBundledAutotuneCache`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteCache`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteCacheBackend`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteCacheJsonSerde`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteCachePassthroughSerde`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteCacheSerde`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteDynamoPGOCache`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`RemoteFxGraphCache`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_CacheStats`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`basis`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`class`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`is`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)

### Functions

- **`__init__`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`__str__`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_backend_get`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_backend_put`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_create_sample`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_decode`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_encode`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_get`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_get_key`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_log_sample`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_put`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`create_cache`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`decode`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`dump_cache_stats`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`encode`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`exception`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`get`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`hit`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`miss`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`put`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)

### Imports

- **`Any`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`Callable`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`_WaitCounter`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`__future__`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`abc`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`abstractmethod`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`annotations`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`atexit`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`collections`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`collections.abc`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`config`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`dataclasses`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`dynamo_timed`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`functools`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`io`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`json`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`logging`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`os`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`override`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`redis`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`rfe.scubadata.scubadata_py3`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`sys`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`torch._dynamo.utils`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`torch._inductor`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`torch._inductor.fb.remote_cache`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`torch.monitor`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`typing`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)
- **`typing_extensions`**: [remote_cache.py_docs.md](./remote_cache.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
