# Documentation: `docs/test/cpp/rpc/test_tensorpipe_serialization.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/rpc/test_tensorpipe_serialization.cpp_docs.md`
- **Size**: 9,356 bytes (9.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/rpc/test_tensorpipe_serialization.cpp`

## File Metadata

- **Path**: `test/cpp/rpc/test_tensorpipe_serialization.cpp`
- **Size**: 7,021 bytes (6.86 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <tensorpipe/common/cpu_buffer.h>
#include <tensorpipe/core/message.h>
#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

TEST(TensorpipeSerialize, Base) {
  // Sender serializes
  at::Tensor t1 = torch::ones({1024}, at::ScalarType::Int);
  at::Tensor t2 = torch::ones({1024}, at::ScalarType::Float);
  std::vector<at::Tensor> tensors{t1, t2};
  std::vector<char> payload = {'1', '2', '3'};
  std::vector<char> payloadCopy = payload; // for testing
  torch::distributed::rpc::MessageType mtype =
      torch::distributed::rpc::MessageType::UNKNOWN;
  int64_t mId = 100;
  auto sendingRpcMessage =
      c10::make_intrusive<torch::distributed::rpc::Message>(
          std::move(payload), std::move(tensors), mtype);
  sendingRpcMessage->setId(mId);
  auto [sendingTpMessage, sendingTpBuffers] =
      torch::distributed::rpc::tensorpipeSerialize(
          std::move(sendingRpcMessage), {}, {});

  // Mimic receiving message descriptor: recvingTpDescriptor is a copy of
  // sendingTpMessage except for the data pointers which are left null.
  tensorpipe::Descriptor recvingTpDescriptor;
  recvingTpDescriptor.metadata = sendingTpMessage.metadata;
  recvingTpDescriptor.payloads.reserve(sendingTpMessage.payloads.size());
  for (auto& tpPayload : sendingTpMessage.payloads) {
    tensorpipe::Descriptor::Payload p;
    p.length = tpPayload.length;
    p.metadata = tpPayload.metadata;
    recvingTpDescriptor.payloads.push_back(std::move(p));
  }
  EXPECT_EQ(
      recvingTpDescriptor.payloads.size(), sendingTpMessage.payloads.size());
  recvingTpDescriptor.tensors.reserve(sendingTpMessage.tensors.size());
  for (auto& tpTensor : sendingTpMessage.tensors) {
    tensorpipe::Descriptor::Tensor t;
    t.length = tpTensor.length;
    t.sourceDevice = tpTensor.buffer.device();
    t.targetDevice = tpTensor.targetDevice;
    t.metadata = tpTensor.metadata;
    recvingTpDescriptor.tensors.push_back(std::move(t));
  }
  EXPECT_EQ(
      recvingTpDescriptor.tensors.size(), sendingTpMessage.tensors.size());

  // Mimic readDescriptor() callback:
  // - Allocate buffers
  // - Fill pointers in tensorpipe message
  auto [recvingTpAllocation, recvingTpBuffers] =
      torch::distributed::rpc::tensorpipeAllocate(recvingTpDescriptor, {});

  // Mimic tensorpipe data transfer
  EXPECT_EQ(
      recvingTpAllocation.payloads.size(), sendingTpMessage.payloads.size());
  for (const auto i : c10::irange(recvingTpAllocation.payloads.size())) {
    tensorpipe::Message::Payload& srcPayload = sendingTpMessage.payloads[i];
    tensorpipe::Allocation::Payload& dstPayload =
        recvingTpAllocation.payloads[i];
    if (srcPayload.length) {
      // Empty vector's data() can return nullptr, use the length to avoid
      // copying into nullptr
      memcpy(dstPayload.data, srcPayload.data, srcPayload.length);
    }
  }
  EXPECT_EQ(
      recvingTpAllocation.tensors.size(), sendingTpMessage.tensors.size());
  for (const auto i : c10::irange(recvingTpAllocation.tensors.size())) {
    tensorpipe::Message::Tensor& srcTensor = sendingTpMessage.tensors[i];
    tensorpipe::Allocation::Tensor& dstTensor = recvingTpAllocation.tensors[i];
    memcpy(
        dstTensor.buffer.unwrap<tensorpipe::CpuBuffer>().ptr,
        srcTensor.buffer.unwrap<tensorpipe::CpuBuffer>().ptr,
        srcTensor.length);
  }

  // Mimic read() callback:
  // - Unpickle
  c10::intrusive_ptr<torch::distributed::rpc::Message> recvingRpcMessage =
      torch::distributed::rpc::tensorpipeDeserialize(
          std::move(recvingTpDescriptor), std::move(recvingTpBuffers));

  // Data is ready
  EXPECT_EQ(mtype, recvingRpcMessage->type());
  EXPECT_EQ(payloadCopy, recvingRpcMessage->payload());
  EXPECT_EQ(mId, recvingRpcMessage->id());
  EXPECT_TRUE(torch::equal(t1, recvingRpcMessage->tensors()[0]));
  EXPECT_TRUE(torch::equal(t2, recvingRpcMessage->tensors()[1]));
}

TEST(TensorpipeSerialize, RecopySparseTensors) {
  // Take a 1K row of a 1M tensors, and make sure we don't send across 1M rows.
  constexpr size_t k1K = 1024;
  at::Tensor main = torch::randn({k1K, k1K});
  at::Tensor tiny = main.select(0, 2); // Select a row in the middle
  EXPECT_EQ(tiny.numel(), k1K);
  EXPECT_EQ(tiny.storage().nbytes() / tiny.itemsize(), k1K * k1K);

  std::vector<at::Tensor> tensors{main, tiny};
  std::vector<char> payload = {'1', '2', '3'};
  torch::distributed::rpc::MessageType mtype =
      torch::distributed::rpc::MessageType::UNKNOWN;
  auto sendingRpcMessage =
      c10::make_intrusive<torch::distributed::rpc::Message>(
          std::move(payload), std::move(tensors), mtype);

  auto [sendingTpMessage, tpBuffers] =
      torch::distributed::rpc::tensorpipeSerialize(
          std::move(sendingRpcMessage), {}, {});

  EXPECT_EQ(tpBuffers.tensors.size(), 2);
  EXPECT_EQ(sendingTpMessage.tensors.size(), 2);
  EXPECT_TRUE(torch::equal(main, tpBuffers.tensors[0]));
  EXPECT_TRUE(torch::equal(tiny, tpBuffers.tensors[1]));
  // Test cloned storage
  EXPECT_EQ(
      main.storage().data(),
      sendingTpMessage.tensors[0].buffer.unwrap<tensorpipe::CpuBuffer>().ptr);
  EXPECT_NE(
      tiny.storage().data(),
      sendingTpMessage.tensors[1].buffer.unwrap<tensorpipe::CpuBuffer>().ptr);
  EXPECT_EQ(tiny.element_size() * k1K, sendingTpMessage.tensors[1].length);
}

TEST(TensorpipeSerialize, NoDeleterTensors) {
  std::vector<float> blob1{.8, .2};
  std::vector<float> blob2{.7, .5, .9};
  at::Tensor t1 = torch::from_blob((float*)(blob1.data()), blob1.size());
  at::Tensor t2 = torch::from_blob((float*)(blob2.data()), blob2.size());
  std::vector<at::Tensor> tensors{t1, t2};
  std::vector<char> payload = {'1', '2', '3'};
  torch::distributed::rpc::MessageType mtype =
      torch::distributed::rpc::MessageType::UNKNOWN;
  auto sendingRpcMessage =
      c10::make_intrusive<torch::distributed::rpc::Message>(
          std::move(payload), std::move(tensors), mtype);

  auto [sendingTpMessage, tpBuffers] =
      torch::distributed::rpc::tensorpipeSerialize(
          std::move(sendingRpcMessage), {}, {});

  EXPECT_EQ(tpBuffers.copiedTensors.size(), 2);
  EXPECT_EQ(sendingTpMessage.tensors.size(), 2);
  EXPECT_EQ(
      tpBuffers.copiedTensors[0].size(), sendingTpMessage.tensors[0].length);
  EXPECT_EQ(
      tpBuffers.copiedTensors[1].size(), sendingTpMessage.tensors[1].length);
  EXPECT_EQ(
      tpBuffers.copiedTensors[0].data(),
      sendingTpMessage.tensors[0].buffer.unwrap<tensorpipe::CpuBuffer>().ptr);
  EXPECT_EQ(
      tpBuffers.copiedTensors[1].data(),
      sendingTpMessage.tensors[1].buffer.unwrap<tensorpipe::CpuBuffer>().ptr);
  EXPECT_TRUE(
      memcmp(
          tpBuffers.copiedTensors[0].data(),
          t1.storage().data(),
          sendingTpMessage.tensors[0].length) == 0);
  EXPECT_TRUE(
      memcmp(
          tpBuffers.copiedTensors[1].data(),
          t2.storage().data(),
          sendingTpMessage.tensors[1].length) == 0);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/rpc`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `c10/util/irange.h`
- `tensorpipe/common/cpu_buffer.h`
- `tensorpipe/core/message.h`
- `torch/csrc/distributed/rpc/tensorpipe_utils.h`
- `torch/torch.h`
- `memory`
- `string`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/rpc/test_tensorpipe_serialization.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/rpc`):

- [`test_wire_serialization.cpp_docs.md`](./test_wire_serialization.cpp_docs.md)
- [`test_e2e_tensorpipe.cpp_docs.md`](./test_e2e_tensorpipe.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`e2e_test_base.h_docs.md`](./e2e_test_base.h_docs.md)
- [`e2e_test_base.cpp_docs.md`](./e2e_test_base.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_tensorpipe_serialization.cpp_docs.md`
- **Keyword Index**: `test_tensorpipe_serialization.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/rpc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/rpc`, which is part of the **testing infrastructure**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/rpc/test_tensorpipe_serialization.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/rpc`):

- [`test_e2e_tensorpipe.cpp_kw.md_docs.md`](./test_e2e_tensorpipe.cpp_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`e2e_test_base.cpp_kw.md_docs.md`](./e2e_test_base.cpp_kw.md_docs.md)
- [`e2e_test_base.h_docs.md_docs.md`](./e2e_test_base.h_docs.md_docs.md)
- [`test_e2e_tensorpipe.cpp_docs.md_docs.md`](./test_e2e_tensorpipe.cpp_docs.md_docs.md)
- [`test_wire_serialization.cpp_docs.md_docs.md`](./test_wire_serialization.cpp_docs.md_docs.md)
- [`test_tensorpipe_serialization.cpp_kw.md_docs.md`](./test_tensorpipe_serialization.cpp_kw.md_docs.md)
- [`e2e_test_base.cpp_docs.md_docs.md`](./e2e_test_base.cpp_docs.md_docs.md)
- [`test_wire_serialization.cpp_kw.md_docs.md`](./test_wire_serialization.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_tensorpipe_serialization.cpp_docs.md_docs.md`
- **Keyword Index**: `test_tensorpipe_serialization.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
