# Documentation: `docs/test/cpp/c10d/ProcessGroupMPITest.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/c10d/ProcessGroupMPITest.cpp_docs.md`
- **Size**: 16,613 bytes (16.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/c10d/ProcessGroupMPITest.cpp`

## File Metadata

- **Path**: `test/cpp/c10d/ProcessGroupMPITest.cpp`
- **Size**: 14,101 bytes (13.77 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <unistd.h>

#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// Wait for work to complete
std::vector<std::vector<at::Tensor>> waitWork(
    const c10::intrusive_ptr<::c10d::ProcessGroupMPI>& pg,
    const std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  std::vector<std::vector<at::Tensor>> outputTensors;
  for (auto& work : works) {
    try {
      work->wait();
    } catch (const std::exception& ex) {
      std::cerr << "Exception received: " << ex.what() << '\n';
      pg->abort();
    }
    outputTensors.emplace_back(work->result());
  }
  return outputTensors;
}

// Wait using Futures
std::vector<std::vector<at::Tensor>> waitFuture(
    const c10::intrusive_ptr<::c10d::ProcessGroupMPI>& pg,
    const std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  std::vector<std::vector<at::Tensor>> outputTensors;
  for (auto& work : works) {
    auto fut = work->getFuture();
    try {
      fut->wait();
    } catch (const std::exception& ex) {
      std::cerr << "Exception received: " << ex.what() << '\n';
      pg->abort();
    }
    auto result = fut->value();
    if (result.isNone()) {
      outputTensors.emplace_back();
    } else if (result.isTensorList()) {
      outputTensors.emplace_back(result.toTensorVector());
    } else {
      TORCH_CHECK(false, "future result should be tensor list or none");
    }
  }
  return outputTensors;
}

void testAllreduce(int iter = 1000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();

  // Generate inputs
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;
  for (const auto i : c10::irange(iter)) {
    auto tensor = at::ones({16, 16}) * i;
    std::vector<at::Tensor> tensors = {tensor};

    // Queue the work.
    c10::intrusive_ptr<::c10d::Work> work = pg->allreduce(tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Get the world size
  auto worldSize = pg->getSize();

  // Verify outputs
  for (const auto i : c10::irange(iter)) {
    const auto expected = worldSize * i;
    auto data = outputTensors[i][0].data_ptr<float>();
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      if (data[j] != static_cast<float>(expected)) {
        TORCH_CHECK(false, "BOOM!");
      }
    }
  }
}

void testBroadcast(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;
  for (const auto i : c10::irange(iter)) {
    auto tensors = std::vector<at::Tensor>();
    if (pg->getRank() == 0) {
      auto tensor = at::ones({16, 16}) * i;
      tensors = std::vector<at::Tensor>({tensor});
    } else {
      auto tensor = at::zeros({16, 16});
      tensors = std::vector<at::Tensor>({tensor});
    }

    // Queue the work.
    c10::intrusive_ptr<::c10d::Work> work = pg->broadcast(tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  for (const auto i : c10::irange(iter)) {
    const auto expected = i;
    auto data = outputTensors[i][0].data_ptr<float>();
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      if (data[j] != static_cast<float>(expected)) {
        TORCH_CHECK(false, "BOOM!");
      }
    }
  }
}

void testReduce(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;
  for (const auto i : c10::irange(iter)) {
    auto tensor = at::ones({16, 16}) * i;
    auto tensors = std::vector<at::Tensor>({tensor});

    // Queue the work.
    c10::intrusive_ptr<::c10d::Work> work = pg->reduce(tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Get the world size
  auto worldSize = pg->getSize();

  if (pg->getRank() == 0) {
    // Verify outputs
    for (const auto i : c10::irange(iter)) {
      const auto expected = worldSize * i;
      auto data = outputTensors[i][0].data_ptr<float>();
      for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
        if (data[j] != static_cast<float>(expected)) {
          TORCH_CHECK(false, "BOOM!");
        }
      }
    }
  }
}

void testAllgather(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (const auto i : c10::irange(iter)) {
    auto tensor = at::ones({16, 16}) * i * rank;
    auto tensors = std::vector<at::Tensor>({tensor});
    auto outputs = std::vector<std::vector<at::Tensor>>(1);
    outputs[0].resize(worldSize);
    for (const auto j : c10::irange(worldSize)) {
      outputs[0][j] = at::zeros({16, 16});
    }

    // Queue the work.
    c10::intrusive_ptr<::c10d::Work> work = pg->allgather(outputs, tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  for (const auto i : c10::irange(iter)) {
    for (const auto j : c10::irange(worldSize)) {
      const auto expected = i * j;
      auto data = outputTensors[i][j].data_ptr<float>();
      for (auto k = 0; k < outputTensors[i][j].numel(); ++k) {
        if (data[k] != static_cast<float>(expected)) {
          TORCH_CHECK(false, "BOOM!");
        }
      }
    }
  }
}

void testAllgatherBase(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (const auto i : c10::irange(iter)) {
    auto tensor = at::ones({16, 16}) * i * rank;
    auto output = at::zeros({worldSize, 16, 16});

    // Queue the work.
    c10::intrusive_ptr<::c10d::Work> work = pg->_allgather_base(output, tensor);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  for (const auto i : c10::irange(iter)) {
    for (const auto j : c10::irange(worldSize)) {
      const auto expected = i * j;
      auto data = outputTensors[i][0][j].data_ptr<float>();
      for (auto k = 0; k < outputTensors[i][0][j].numel(); ++k) {
        if (data[k] != static_cast<float>(expected)) {
          TORCH_CHECK(false, "BOOM!");
        }
      }
    }
  }
}

void testReduceScatter(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  int count = 2;
  for (const auto i : c10::irange(iter)) {
    auto tensors = std::vector<std::vector<at::Tensor>>(1);
    tensors[0].resize(worldSize);
    for (const auto j : c10::irange(worldSize)) {
      tensors[0][j] = at::ones({count, count}) * i * rank;
    }
    auto output = at::zeros({count, count});
    auto outputs = std::vector<at::Tensor>({output});

    // Queue the work.
    c10::intrusive_ptr<::c10d::Work> work =
        pg->reduce_scatter(outputs, tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  for (const auto i : c10::irange(iter)) {
    const auto expected = i * (worldSize * (worldSize - 1)) / 2.0;
    auto data = outputTensors[i][0].data_ptr<float>();
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      if (data[j] != static_cast<float>(expected)) {
        TORCH_CHECK(false, "BOOM!");
      }
    }
  }
}

void testReduceScatterBase(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (const auto i : c10::irange(iter)) {
    auto tensor = at::ones({worldSize, 16, 16}) * i * rank;
    auto output = at::zeros({16, 16});
    auto outputs = std::vector<at::Tensor>({output});

    // Queue the work.
    c10::intrusive_ptr<::c10d::Work> work =
        pg->_reduce_scatter_base(output, tensor);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  for (const auto i : c10::irange(iter)) {
    const auto expected = i * (worldSize * (worldSize - 1)) / 2.0;
    auto data = outputTensors[i][0].data_ptr<float>();
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      if (data[j] != static_cast<float>(expected)) {
        TORCH_CHECK(false, "BOOM!");
      }
    }
  }
}

void testGather(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (const auto i : c10::irange(iter)) {
    auto tensor = at::ones({16, 16}) * i * rank;
    auto tensors = std::vector<at::Tensor>({tensor});
    auto outputs = std::vector<std::vector<at::Tensor>>(0);
    if (rank == 0) {
      outputs = std::vector<std::vector<at::Tensor>>(1);
      outputs[0].resize(worldSize);
      for (const auto j : c10::irange(worldSize)) {
        outputs[0][j] = at::zeros({16, 16});
      }
    }

    // Queue the work.
    c10::intrusive_ptr<::c10d::Work> work = pg->gather(outputs, tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  if (rank == 0) {
    for (const auto i : c10::irange(iter)) {
      for (const auto j : c10::irange(worldSize)) {
        const auto expected = i * j;
        auto data = outputTensors[i][j].data_ptr<float>();
        for (auto k = 0; k < outputTensors[i][j].numel(); ++k) {
          if (data[k] != static_cast<float>(expected)) {
            TORCH_CHECK(false, "BOOM!");
          }
        }
      }
    }
  } else {
    for (const auto i : c10::irange(iter)) {
      if (!outputTensors[i].empty()) {
        TORCH_CHECK(false, "BOOM!");
      }
    }
  }
}

void testScatter(int iter = 1) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (const auto i : c10::irange(iter)) {
    auto tensor = at::zeros({16, 16});
    auto tensors = std::vector<at::Tensor>({tensor});
    auto inputs = std::vector<std::vector<at::Tensor>>(0);
    if (rank == 0) {
      inputs = std::vector<std::vector<at::Tensor>>(1);
      inputs[0].resize(worldSize);
      for (const auto j : c10::irange(worldSize)) {
        inputs[0][j] = at::ones({16, 16}) * i * j;
      }
    }

    // Queue the work.
    c10::intrusive_ptr<::c10d::Work> work = pg->scatter(tensors, inputs);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  for (const auto i : c10::irange(iter)) {
    for (const auto j : c10::irange(worldSize)) {
      const auto expected = i * j;
      auto data = outputTensors[i][0].data_ptr<float>();
      for (auto k = 0; k < outputTensors[i][0].numel(); ++k) {
        if (data[k] != static_cast<float>(expected)) {
          TORCH_CHECK(false, "BOOM!");
        }
      }
    }
  }
}

void testSendRecv(bool recvAnysource, int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // Generate inputs
  std::vector<c10::intrusive_ptr<::c10d::Work>> works;

  // pg->send does not keep sent tensors alive, so we need to.
  std::vector<std::vector<at::Tensor>> sendTensors(iter);
  auto rank = pg->getRank();
  for (const auto i : c10::irange(iter)) {
    if (rank == 0) {
      auto tensor = at::ones({16, 16}) * i;
      sendTensors[i] = std::vector<at::Tensor>({tensor});

      // Queue the work.
      c10::intrusive_ptr<::c10d::Work> work = pg->send(sendTensors[i], 1, 0);
      works.push_back(std::move(work));
    } else {
      auto tensor = at::zeros({16, 16});
      auto recvTensors = std::vector<at::Tensor>({tensor});

      // Queue the work.
      if (!recvAnysource) {
        c10::intrusive_ptr<::c10d::Work> work = pg->recv(recvTensors, 0, 0);
        works.push_back(std::move(work));
      } else {
        c10::intrusive_ptr<::c10d::Work> work =
            pg->recvAnysource(recvTensors, 0);
        works.push_back(std::move(work));
      }
    }
  }

  auto outputTensors = waitWork(pg, works);
  if (rank == 0) {
    return;
  }

  std::vector<int> srcRanks;
  if (recvAnysource) {
    for (const auto& work : works) {
      srcRanks.push_back(work->sourceRank());
    }
  }

  // Verify outputs
  for (const auto i : c10::irange(iter)) {
    if (recvAnysource && srcRanks[i] != 0) {
      TORCH_CHECK(false, "src rank is wrong for recvAnysource");
    }
    const auto expected = i;
    auto data = outputTensors[i][0].data_ptr<float>();
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      if (data[j] != static_cast<float>(expected)) {
        TORCH_CHECK(false, "BOOM!");
      }
    }
  }
}

void testBackendName() {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  if (pg->getBackendName() != std::string(c10d::MPI_BACKEND_NAME)) {
    TORCH_CHECK(false, "BOOM!");
  }
}

int main(int argc, char** argv) {
#ifdef MPIEXEC
  // If we are within an openmpi mpirun, then skip the exec
  if (!std::getenv("OMPI_COMM_WORLD_SIZE")) {
    std::cout << "Execute mpiexec from: " << STR(MPIEXEC) << '\n';
    execl(STR(MPIEXEC), "-np 2", argv[0], (char*)nullptr);
  }

  testAllreduce();
  testBroadcast();
  testReduce();
  testAllgather();
  testAllgatherBase();
  testReduceScatter();
  testReduceScatterBase();
  testGather();
  testScatter();
  testSendRecv(false);
  testSendRecv(true);
  testBackendName();

  std::cout << "Test successful" << '\n';
#else
  std::cout << "MPI executable not found, skipping test" << std::endl;
#endif
  return EXIT_SUCCESS;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 31 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/c10d`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `unistd.h`
- `c10/util/irange.h`
- `torch/csrc/distributed/c10d/ProcessGroupMPI.hpp`
- `cstdlib`
- `iostream`
- `string`


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

This is a test file. Run it with:

```bash
python test/cpp/c10d/ProcessGroupMPITest.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/c10d`):

- [`TCPStoreTest.cpp_docs.md`](./TCPStoreTest.cpp_docs.md)
- [`ProcessGroupUCCTest.cpp_docs.md`](./ProcessGroupUCCTest.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`ProcessGroupNCCLErrorsTest.cpp_docs.md`](./ProcessGroupNCCLErrorsTest.cpp_docs.md)
- [`FileStoreTest.cpp_docs.md`](./FileStoreTest.cpp_docs.md)
- [`ProcessGroupNCCLTest.cpp_docs.md`](./ProcessGroupNCCLTest.cpp_docs.md)
- [`HashStoreTest.cpp_docs.md`](./HashStoreTest.cpp_docs.md)
- [`TestUtils.hpp_docs.md`](./TestUtils.hpp_docs.md)
- [`StoreTestCommon.hpp_docs.md`](./StoreTestCommon.hpp_docs.md)


## Cross-References

- **File Documentation**: `ProcessGroupMPITest.cpp_docs.md`
- **Keyword Index**: `ProcessGroupMPITest.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/c10d`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/c10d`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/c10d/ProcessGroupMPITest.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/c10d`):

- [`FileStoreTest.cpp_kw.md_docs.md`](./FileStoreTest.cpp_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`ProcessGroupUCCTest.cpp_docs.md_docs.md`](./ProcessGroupUCCTest.cpp_docs.md_docs.md)
- [`ProcessGroupNCCLTest.cpp_docs.md_docs.md`](./ProcessGroupNCCLTest.cpp_docs.md_docs.md)
- [`HashStoreTest.cpp_kw.md_docs.md`](./HashStoreTest.cpp_kw.md_docs.md)
- [`CUDATest.hpp_docs.md_docs.md`](./CUDATest.hpp_docs.md_docs.md)
- [`ProcessGroupNCCLErrorsTest.cpp_kw.md_docs.md`](./ProcessGroupNCCLErrorsTest.cpp_kw.md_docs.md)
- [`HashStoreTest.cpp_docs.md_docs.md`](./HashStoreTest.cpp_docs.md_docs.md)
- [`CUDATest.hpp_kw.md_docs.md`](./CUDATest.hpp_kw.md_docs.md)
- [`StoreTestCommon.hpp_docs.md_docs.md`](./StoreTestCommon.hpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ProcessGroupMPITest.cpp_docs.md_docs.md`
- **Keyword Index**: `ProcessGroupMPITest.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
