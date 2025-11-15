# Documentation: `third_party/miniz-3.0.2/examples/example1.c`

## File Metadata

- **Path**: `third_party/miniz-3.0.2/examples/example1.c`
- **Size**: 3,109 bytes (3.04 KB)
- **Type**: Source File (.c)
- **Extension**: `.c`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```c
// example1.c - Demonstrates miniz.c's compress() and uncompress() functions (same as zlib's).
// Public domain, May 15 2011, Rich Geldreich, richgel99@gmail.com. See "unlicense" statement at the end of tinfl.c.
#include <stdio.h>
#include "miniz.h"
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;

// The string to compress.
static const char *s_pStr = "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson." \
  "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson." \
  "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson." \
  "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson." \
  "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson." \
  "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson." \
  "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson.";

int main(int argc, char *argv[])
{
  uint step = 0;
  int cmp_status;
  uLong src_len = (uLong)strlen(s_pStr);
  uLong cmp_len = compressBound(src_len);
  uLong uncomp_len = src_len;
  uint8 *pCmp, *pUncomp;
  uint total_succeeded = 0;
  (void)argc, (void)argv;

  printf("miniz.c version: %s\n", MZ_VERSION);

  do
  {
    // Allocate buffers to hold compressed and uncompressed data.
    pCmp = (mz_uint8 *)malloc((size_t)cmp_len);
    pUncomp = (mz_uint8 *)malloc((size_t)src_len);
    if ((!pCmp) || (!pUncomp))
    {
      printf("Out of memory!\n");
      return EXIT_FAILURE;
    }

    // Compress the string.
    cmp_status = compress(pCmp, &cmp_len, (const unsigned char *)s_pStr, src_len);
    if (cmp_status != Z_OK)
    {
      printf("compress() failed!\n");
      free(pCmp);
      free(pUncomp);
      return EXIT_FAILURE;
    }

    printf("Compressed from %u to %u bytes\n", (mz_uint32)src_len, (mz_uint32)cmp_len);

    if (step)
    {
      // Purposely corrupt the compressed data if fuzzy testing (this is a very crude fuzzy test).
      uint n = 1 + (rand() % 3);
      while (n--)
      {
        uint i = rand() % cmp_len;
        pCmp[i] ^= (rand() & 0xFF);
      }
    }

    // Decompress.
    cmp_status = uncompress(pUncomp, &uncomp_len, pCmp, cmp_len);
    total_succeeded += (cmp_status == Z_OK);

    if (step)
    {
      printf("Simple fuzzy test: step %u total_succeeded: %u\n", step, total_succeeded);
    }
    else
    {
      if (cmp_status != Z_OK)
      {
        printf("uncompress failed!\n");
        free(pCmp);
        free(pUncomp);
        return EXIT_FAILURE;
      }

      printf("Decompressed from %u to %u bytes\n", (mz_uint32)cmp_len, (mz_uint32)uncomp_len);

      // Ensure uncompress() returned the expected data.
      if ((uncomp_len != src_len) || (memcmp(pUncomp, s_pStr, (size_t)src_len)))
      {
        printf("Decompression failed!\n");
        free(pCmp);
        free(pUncomp);
        return EXIT_FAILURE;
      }
    }

    free(pCmp);
    free(pUncomp);

    step++;

    // Keep on fuzzy testing if there's a non-empty command line.
  } while (argc >= 2);

  printf("Success.\n");
  return EXIT_SUCCESS;
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `third_party/miniz-3.0.2/examples`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `third_party/miniz-3.0.2/examples`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`third_party/miniz-3.0.2/examples`):

- [`example2.c_docs.md`](./example2.c_docs.md)
- [`example4.c_docs.md`](./example4.c_docs.md)
- [`example3.c_docs.md`](./example3.c_docs.md)
- [`example6.c_docs.md`](./example6.c_docs.md)
- [`example5.c_docs.md`](./example5.c_docs.md)


## Cross-References

- **File Documentation**: `example1.c_docs.md`
- **Keyword Index**: `example1.c_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
