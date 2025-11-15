# Documentation: `docs/test/cpp/nativert/test_file_util.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/nativert/test_file_util.cpp_docs.md`
- **Size**: 4,923 bytes (4.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/nativert/test_file_util.cpp`

## File Metadata

- **Path**: `test/cpp/nativert/test_file_util.cpp`
- **Size**: 2,302 bytes (2.25 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <torch/nativert/common/FileUtil.h>
#include <fstream>

namespace torch {
namespace nativert {

TEST(FileUtilTest, OpenNoInt) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  int fd = openNoInt("tmp_file.txt", O_RDONLY, 0);
  ASSERT_GE(fd, 0);

  closeNoInt(fd);
}

TEST(FileUtilTest, CloseNoInt) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  int fd = openNoInt("tmp_file.txt", O_RDONLY, 0);
  ASSERT_GE(fd, 0);

  int result = closeNoInt(fd);
  ASSERT_EQ(result, 0);
}

TEST(FileUtilTest, WriteFull) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  int fd = openNoInt("tmp_file.txt", O_WRONLY | O_CREAT, 0644);
  ASSERT_GE(fd, 0);

  const char* data = "Hello, World!";
  ssize_t bytesWritten = writeFull(fd, data, strlen(data));
  ASSERT_EQ(bytesWritten, strlen(data));

  closeNoInt(fd);
}

TEST(FileUtilTest, ReadFull) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile << "Hello, World!";
  tmpFile.close();

  int fd = openNoInt("tmp_file.txt", O_RDONLY, 0);
  ASSERT_GE(fd, 0);

  char buffer[1024];
  ssize_t bytesRead = readFull(fd, buffer, 1024);
  ASSERT_EQ(bytesRead, 13); // length of "Hello, World!"

  closeNoInt(fd);
}

TEST(FileUtilTest, FileConstructor) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  File file("tmp_file.txt", O_RDONLY, 0);
  ASSERT_GE(file.fd(), 0);

  file.close();
}

TEST(FileUtilTest, FileMoveConstructor) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  File file1("tmp_file.txt", O_RDONLY, 0);
  File file2(std::move(file1));

  ASSERT_GE(file2.fd(), 0);
  ASSERT_EQ(file1.fd(), -1);

  file2.close();
}

TEST(FileUtilTest, FileAssignmentOperator) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  File file1("tmp_file.txt", O_RDONLY, 0);
  File file2;

  file2 = std::move(file1);

  ASSERT_GE(file2.fd(), 0);
  ASSERT_EQ(file1.fd(), -1);

  file2.close();
}

TEST(FileUtilTest, TemporaryFile) {
  File file = File::temporary();
  ASSERT_GE(file.fd(), 0);

  file.close();
}

} // namespace nativert
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `nativert`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/nativert`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/nativert/common/FileUtil.h`
- `fstream`


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
python test/cpp/nativert/test_file_util.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/nativert`):

- [`test_alias_analyzer.cpp_docs.md`](./test_alias_analyzer.cpp_docs.md)
- [`test_placement.cpp_docs.md`](./test_placement.cpp_docs.md)
- [`test_static_kernel_ops.cpp_docs.md`](./test_static_kernel_ops.cpp_docs.md)
- [`test_static_dispatch_kernel_registration.cpp_docs.md`](./test_static_dispatch_kernel_registration.cpp_docs.md)
- [`test_graph.cpp_docs.md`](./test_graph.cpp_docs.md)
- [`test_c10_kernel.cpp_docs.md`](./test_c10_kernel.cpp_docs.md)
- [`test_function_schema.cpp_docs.md`](./test_function_schema.cpp_docs.md)
- [`test_execution_frame.cpp_docs.md`](./test_execution_frame.cpp_docs.md)
- [`test_triton_kernel_manager_registration.cpp_docs.md`](./test_triton_kernel_manager_registration.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)


## Cross-References

- **File Documentation**: `test_file_util.cpp_docs.md`
- **Keyword Index**: `test_file_util.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/nativert`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/nativert`, which is part of the **testing infrastructure**.



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
python docs/test/cpp/nativert/test_file_util.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/nativert`):

- [`test_execution_frame.cpp_kw.md_docs.md`](./test_execution_frame.cpp_kw.md_docs.md)
- [`test_tensor_meta.cpp_kw.md_docs.md`](./test_tensor_meta.cpp_kw.md_docs.md)
- [`test_graph_signature.cpp_kw.md_docs.md`](./test_graph_signature.cpp_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`test_static_kernel_ops.cpp_kw.md_docs.md`](./test_static_kernel_ops.cpp_kw.md_docs.md)
- [`test_layout_planner_algorithm.cpp_docs.md_docs.md`](./test_layout_planner_algorithm.cpp_docs.md_docs.md)
- [`test_pass_manager.cpp_docs.md_docs.md`](./test_pass_manager.cpp_docs.md_docs.md)
- [`test_static_dispatch_kernel_registration.cpp_kw.md_docs.md`](./test_static_dispatch_kernel_registration.cpp_kw.md_docs.md)
- [`test_placement.cpp_kw.md_docs.md`](./test_placement.cpp_kw.md_docs.md)
- [`test_static_kernel_ops.cpp_docs.md_docs.md`](./test_static_kernel_ops.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_file_util.cpp_docs.md_docs.md`
- **Keyword Index**: `test_file_util.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
