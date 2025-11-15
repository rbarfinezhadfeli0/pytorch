# Documentation: `test/cpp/jit/test_lexer.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_lexer.cpp`
- **Size**: 3,070 bytes (3.00 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <torch/csrc/jit/frontend/lexer.h>

namespace torch::jit {

TEST(LexerTest, AllTokens) {
  std::vector<std::pair<int /* TokenKind */, std::string>> tokens;
  for (const char* ch = valid_single_char_tokens; *ch; ch++) {
    tokens.emplace_back(*ch, std::string(1, *ch));
  }
#define ADD_TOKEN(tok, _, tokstring)     \
  if (*tokstring) {                      \
    tokens.emplace_back(tok, tokstring); \
  }
  TC_FORALL_TOKEN_KINDS(ADD_TOKEN);
#undef ADD_TOKEN

  for (const auto& [kind, token] : tokens) {
    Lexer l(std::make_shared<Source>(token));
    const auto& tok = l.cur();
    EXPECT_EQ(kind, tok.kind) << tok.range.text().str();
    EXPECT_EQ(token, tok.range.text().str()) << tok.range.text().str();
    l.next();
    EXPECT_EQ(l.cur().kind, TK_EOF);
  }
}

TEST(LexerTest, SlightlyOffIsNot) {
  std::vector<std::string> suffixes = {"", " ", "**"};
  for (const auto& suffix : suffixes) {
    std::vector<std::string> extras = {"n", "no", "no3"};
    for (const auto& extra : extras) {
      std::string s = "is " + extra + suffix;
      Lexer l(std::make_shared<Source>(s));
      const auto& is_tok = l.next();
      EXPECT_EQ(is_tok.kind, TK_IS) << is_tok.range.text().str();
      const auto& no_tok = l.cur();
      EXPECT_EQ(no_tok.kind, TK_IDENT) << no_tok.range.text().str();
      EXPECT_EQ(no_tok.range.text().str(), extra) << no_tok.range.text().str();
    }
  }
}

TEST(LexerTest, SlightlyOffNotIn) {
  std::vector<std::string> suffixes = {"", " ", "**"};
  for (const auto& suffix : suffixes) {
    std::vector<std::string> extras = {"i", "i3"};
    for (const auto& extra : extras) {
      std::string s = "not " + extra + suffix;
      Lexer l(std::make_shared<Source>(s));
      const auto& not_tok = l.next();
      EXPECT_EQ(not_tok.kind, TK_NOT) << not_tok.range.text().str();
      const auto& in_tok = l.cur();
      EXPECT_EQ(in_tok.kind, TK_IDENT) << in_tok.range.text().str();
      EXPECT_EQ(in_tok.range.text().str(), extra) << in_tok.range.text().str();
    }
  }
}

TEST(LexerTest, IsNoteBug) {
  // The code string `is note` is lexed as TK_ISNOT followed by a
  // TK_IDENT that is an e. This is not how it works in Python, but
  // presumably we need to maintain this behavior.
  Lexer l(std::make_shared<Source>("is note"));
  const auto is_not_tok = l.next();
  EXPECT_EQ(is_not_tok.kind, TK_ISNOT);
  const auto e_tok = l.next();
  EXPECT_EQ(e_tok.kind, TK_IDENT);
  EXPECT_EQ(e_tok.range.text(), "e");
  const auto eof_tok = l.next();
  EXPECT_EQ(eof_tok.kind, TK_EOF);
}

TEST(LexerTest, NotInpBug) {
  // Another manifestation of the above IsNoteBug; `not inp` is lexed
  // as TK_NOT_IN followed by a TK_IDENT that is a p. Again, not how
  // it works in Python.
  Lexer l(std::make_shared<Source>("not inp"));
  const auto not_in_tok = l.next();
  EXPECT_EQ(not_in_tok.kind, TK_NOTIN);
  const auto p_tok = l.next();
  EXPECT_EQ(p_tok.kind, TK_IDENT);
  EXPECT_EQ(p_tok.range.text(), "p");
  const auto eof_tok = l.next();
  EXPECT_EQ(eof_tok.kind, TK_EOF);
}
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/csrc/jit/frontend/lexer.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/jit/test_lexer.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/jit`):

- [`test_code_template.cpp_docs.md`](./test_code_template.cpp_docs.md)
- [`test_memory_dag.cpp_docs.md`](./test_memory_dag.cpp_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_cleanup_passes.cpp_docs.md`](./test_cleanup_passes.cpp_docs.md)
- [`test_union.cpp_docs.md`](./test_union.cpp_docs.md)
- [`test_subgraph_rewriter.cpp_docs.md`](./test_subgraph_rewriter.cpp_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md)
- [`test_lite_interpreter_direct.cpp_docs.md`](./test_lite_interpreter_direct.cpp_docs.md)
- [`test_save_load.cpp_docs.md`](./test_save_load.cpp_docs.md)
- [`test_module_api.cpp_docs.md`](./test_module_api.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_lexer.cpp_docs.md`
- **Keyword Index**: `test_lexer.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
