# Documentation: `docs/caffe2/utils/string_utils.cc_docs.md`

## File Metadata

- **Path**: `docs/caffe2/utils/string_utils.cc_docs.md`
- **Size**: 5,067 bytes (4.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `caffe2/utils/string_utils.cc`

## File Metadata

- **Path**: `caffe2/utils/string_utils.cc`
- **Size**: 3,015 bytes (2.94 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include "caffe2/utils/string_utils.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <cstdint>

namespace caffe2 {

std::vector<std::string>
split(char separator, const std::string& string, bool ignore_empty) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (getline(ss, item, separator)) {
    if (!ignore_empty || !item.empty()) {
      pieces.push_back(std::move(item));
    }
  }
  return pieces;
}

std::string trim(const std::string& str) {
  size_t left = str.find_first_not_of(' ');
  if (left == std::string::npos) {
    return str;
  }
  size_t right = str.find_last_not_of(' ');
  return str.substr(left, (right - left + 1));
}

size_t editDistance(
  const std::string& s1, const std::string& s2, size_t max_distance)
  {
    std::vector<size_t> current(s1.length() + 1);
    std::vector<size_t> previous(s1.length() + 1);
    std::vector<size_t> previous1(s1.length() + 1);

    return editDistanceHelper(
        s1.c_str(),
        s1.length(),
        s2.c_str(),
        s2.length(),
        current,
        previous,
        previous1,
        max_distance
    );
  }
  #define NEXT_UNSAFE(s, i, c) { \
      (c)=(uint8_t)(s)[(i)++]; \
  }

int32_t editDistanceHelper(const char* s1,
  size_t s1_len,
  const char* s2,
  size_t s2_len,
  std::vector<size_t> &current,
  std::vector<size_t> &previous,
  std::vector<size_t> &previous1,
  size_t max_distance) {
    if (max_distance) {
      if (std::max(s1_len, s2_len) - std::min(s1_len, s2_len) > max_distance) {
        return max_distance+1;
      }
    }

    for (size_t j = 0; j <= s1_len; ++j) {
      current[j] = j;
    }

    int32_t str2_offset = 0;
    char prev2 = 0;
    for (size_t i = 1; i <= s2_len; ++i) {
      swap(previous1, previous);
      swap(current, previous);
      current[0] = i;

      // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
      char c2 = s2[str2_offset];
      char prev1 = 0;
      int32_t str1_offset = 0;

      NEXT_UNSAFE(s2, str2_offset, c2);

      size_t current_min = s1_len;
      for (size_t j = 1; j <= s1_len; ++j) {
        size_t insertion = previous[j] + 1;
        size_t deletion = current[j - 1] + 1;
        size_t substitution = previous[j - 1];
        size_t transposition = insertion;
        // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
        char c1 = s1[str1_offset];

        NEXT_UNSAFE(s1, str1_offset, c1);

        if (c1 != c2) {
          substitution += 1;
        }


        if (prev1 == c2 && prev2 == c1 && j > 1 && i > 1) {
          transposition = previous1[j - 2] + 1;
        }
        prev1 = c1;

        current[j] = std::min(std::min(insertion, deletion),
                         std::min(substitution, transposition));
        current_min = std::min(current_min, current[j]);
      }


      if (max_distance != 0 && current_min > max_distance) {
        return max_distance+1;
      }

      prev2 = c2;
    }

    return current[s1_len];
  }
} // namespace caffe2

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `caffe2`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `caffe2/utils`, which is part of the **Caffe2** deep learning framework.



## Dependencies

### Import Dependencies

This file includes:

- `caffe2/utils/string_utils.h`
- `algorithm`
- `sstream`
- `vector`
- `cstdint`


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

Files in the same folder (`caffe2/utils`):

- [`proto_wrap.cc_docs.md`](./proto_wrap.cc_docs.md)
- [`fixed_divisor.h_docs.md`](./fixed_divisor.h_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`proto_wrap.h_docs.md`](./proto_wrap.h_docs.md)
- [`string_utils.h_docs.md`](./string_utils.h_docs.md)


## Cross-References

- **File Documentation**: `string_utils.cc_docs.md`
- **Keyword Index**: `string_utils.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/caffe2/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/caffe2/utils`, which is part of the **Caffe2** deep learning framework.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/caffe2/utils`):

- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`proto_wrap.h_docs.md_docs.md`](./proto_wrap.h_docs.md_docs.md)
- [`string_utils.h_docs.md_docs.md`](./string_utils.h_docs.md_docs.md)
- [`string_utils.h_kw.md_docs.md`](./string_utils.h_kw.md_docs.md)
- [`proto_wrap.h_kw.md_docs.md`](./proto_wrap.h_kw.md_docs.md)
- [`string_utils.cc_kw.md_docs.md`](./string_utils.cc_kw.md_docs.md)
- [`fixed_divisor.h_kw.md_docs.md`](./fixed_divisor.h_kw.md_docs.md)
- [`proto_wrap.cc_docs.md_docs.md`](./proto_wrap.cc_docs.md_docs.md)
- [`CMakeLists.txt_kw.md_docs.md`](./CMakeLists.txt_kw.md_docs.md)


## Cross-References

- **File Documentation**: `string_utils.cc_docs.md_docs.md`
- **Keyword Index**: `string_utils.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
