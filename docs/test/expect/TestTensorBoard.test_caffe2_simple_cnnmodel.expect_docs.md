# Documentation: TestTensorBoard.test_caffe2_simple_cnnmodel.expect

## File Metadata
- **Path**: `test/expect/TestTensorBoard.test_caffe2_simple_cnnmodel.expect`
- **Size**: 4468 bytes
- **Lines**: 319
- **Extension**: .expect
- **Type**: Regular file

## Original Source

```expect
node {
  name: "conv1/XavierFill"
  op: "XavierFill"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 96
          }
          dim {
            size: 3
          }
          dim {
            size: 11
          }
          dim {
            size: 11
          }
        }
      }
    }
  }
}
node {
  name: "conv1/ConstantFill"
  op: "ConstantFill"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 96
          }
        }
      }
    }
  }
}
node {
  name: "classifier/XavierFill"
  op: "XavierFill"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 1000
          }
          dim {
            size: 4096
          }
        }
      }
    }
  }
}
node {
  name: "classifier/ConstantFill"
  op: "ConstantFill"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 1000
          }
        }
      }
    }
  }
}
node {
  name: "conv1/Conv"
  op: "Conv"
  input: "conv1/data"
  input: "conv1/conv1_w"
  input: "conv1/conv1_b"
  attr {
    key: "exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 11
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 4
    }
  }
}
node {
  name: "conv1/Relu"
  op: "Relu"
  input: "conv1/conv1"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "conv1/MaxPool"
  op: "MaxPool"
  input: "conv1/conv1_1"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 2
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 2
    }
  }
}
node {
  name: "classifier/FC"
  op: "FC"
  input: "conv1/pool1"
  input: "classifier/fc_w"
  input: "classifier/fc_b"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "classifier/Softmax"
  op: "Softmax"
  input: "classifier/fc"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "classifier/LabelCrossEntropy"
  op: "LabelCrossEntropy"
  input: "classifier/pred"
  input: "classifier/label"
}
node {
  name: "classifier/AveragedLoss"
  op: "AveragedLoss"
  input: "classifier/xent"
}
node {
  name: "conv1/conv1_w"
  op: "Blob"
  input: "conv1/XavierFill:0"
}
node {
  name: "conv1/conv1_b"
  op: "Blob"
  input: "conv1/ConstantFill:0"
}
node {
  name: "classifier/fc_w"
  op: "Blob"
  input: "classifier/XavierFill:0"
}
node {
  name: "classifier/fc_b"
  op: "Blob"
  input: "classifier/ConstantFill:0"
}
node {
  name: "conv1/data"
  op: "Placeholder"
}
node {
  name: "conv1/conv1_w"
  op: "Blob"
  input: "conv1/XavierFill:0"
}
node {
  name: "conv1/conv1_b"
  op: "Blob"
  input: "conv1/ConstantFill:0"
}
node {
  name: "conv1/conv1"
  op: "Blob"
  input: "conv1/Conv:0"
}
node {
  name: "conv1/conv1"
  op: "Blob"
  input: "conv1/Conv:0"
}
node {
  name: "conv1/conv1_1"
  op: "Blob"
  input: "conv1/Relu:0"
}
node {
  name: "conv1/conv1_1"
  op: "Blob"
  input: "conv1/Relu:0"
}
node {
  name: "conv1/pool1"
  op: "Blob"
  input: "conv1/MaxPool:0"
}
node {
  name: "conv1/pool1"
  op: "Blob"
  input: "conv1/MaxPool:0"
}
node {
  name: "classifier/fc_w"
  op: "Blob"
  input: "classifier/XavierFill:0"
}
node {
  name: "classifier/fc_b"
  op: "Blob"
  input: "classifier/ConstantFill:0"
}
node {
  name: "classifier/fc"
  op: "Blob"
  input: "classifier/FC:0"
}
node {
  name: "classifier/fc"
  op: "Blob"
  input: "classifier/FC:0"
}
node {
  name: "classifier/pred"
  op: "Blob"
  input: "classifier/Softmax:0"
}
node {
  name: "classifier/pred"
  op: "Blob"
  input: "classifier/Softmax:0"
}
node {
  name: "classifier/label"
  op: "Placeholder"
}
node {
  name: "classifier/xent"
  op: "Blob"
  input: "classifier/LabelCrossEntropy:0"
}
node {
  name: "classifier/xent"
  op: "Blob"
  input: "classifier/LabelCrossEntropy:0"
}
node {
  name: "classifier/loss"
  op: "Blob"
  input: "classifier/AveragedLoss:0"
}

```

## High-Level Overview

This file is part of the PyTorch repository. It is a source or configuration file.

## Detailed Walkthrough


## Key Components

The file contains 550 words across 319 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4468 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
