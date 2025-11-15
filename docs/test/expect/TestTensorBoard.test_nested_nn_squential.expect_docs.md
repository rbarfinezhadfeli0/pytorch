# Documentation: TestTensorBoard.test_nested_nn_squential.expect

## File Metadata
- **Path**: `test/expect/TestTensorBoard.test_nested_nn_squential.expect`
- **Size**: 6865 bytes
- **Lines**: 254
- **Extension**: .expect
- **Type**: Regular file

## Original Source

```expect
node {
  name: "input/x"
  op: "IO Node"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
      }
    }
  }
  attr {
    key: "attr"
    value {
      s: ""
    }
  }
}
node {
  name: "output/output.1"
  op: "IO Node"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[1]/98"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
      }
    }
  }
  attr {
    key: "attr"
    value {
      s: ""
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[0]/bias/bias.1"
  op: "prim::GetAttr"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[0]/weight/_0.1"
  attr {
    key: "attr"
    value {
      s: "{ name :  bias }"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[0]/weight/weight.1"
  op: "prim::GetAttr"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[0]/weight/_0.1"
  attr {
    key: "attr"
    value {
      s: "{ name :  weight }"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[0]/input.1"
  op: "aten::linear"
  input: "input/x"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[0]/weight/weight.1"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[0]/bias/bias.1"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 2
          }
          dim {
            size: 4
          }
        }
      }
    }
  }
  attr {
    key: "attr"
    value {
      s: "{}"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[1]/bias/bias.3"
  op: "prim::GetAttr"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[1]/weight/_1.1"
  attr {
    key: "attr"
    value {
      s: "{ name :  bias }"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[1]/weight/weight.3"
  op: "prim::GetAttr"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[1]/weight/_1.1"
  attr {
    key: "attr"
    value {
      s: "{ name :  weight }"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[1]/input.3"
  op: "aten::linear"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[0]/input.1"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[1]/weight/weight.3"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[1]/bias/bias.3"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
      }
    }
  }
  attr {
    key: "attr"
    value {
      s: "{}"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[0]/bias/bias.5"
  op: "prim::GetAttr"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[0]/weight/_0"
  attr {
    key: "attr"
    value {
      s: "{ name :  bias }"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[0]/weight/weight.5"
  op: "prim::GetAttr"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[0]/weight/_0"
  attr {
    key: "attr"
    value {
      s: "{ name :  weight }"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[0]/input"
  op: "aten::linear"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[0]/Sequential[inner_nn_squential]/Linear[1]/input.3"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[0]/weight/weight.5"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[0]/bias/bias.5"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 2
          }
          dim {
            size: 4
          }
        }
      }
    }
  }
  attr {
    key: "attr"
    value {
      s: "{}"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[1]/bias/bias"
  op: "prim::GetAttr"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[1]/weight/_1.3"
  attr {
    key: "attr"
    value {
      s: "{ name :  bias }"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[1]/weight/weight"
  op: "prim::GetAttr"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[1]/weight/_1.3"
  attr {
    key: "attr"
    value {
      s: "{ name :  weight }"
    }
  }
}
node {
  name: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[1]/98"
  op: "aten::linear"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[0]/input"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[1]/weight/weight"
  input: "OuterNNSquential/Sequential[outer_nn_squential]/InnerNNSquential[1]/Sequential[inner_nn_squential]/Linear[1]/bias/bias"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
      }
    }
  }
  attr {
    key: "attr"
    value {
      s: "{}"
    }
  }
}
versions {
  producer: 22
}

```

## High-Level Overview

This file is part of the PyTorch repository. It is a source or configuration file.

## Detailed Walkthrough


## Key Components

The file contains 463 words across 254 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 6865 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
