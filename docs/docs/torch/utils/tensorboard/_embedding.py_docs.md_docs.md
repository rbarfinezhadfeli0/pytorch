# Documentation: `docs/torch/utils/tensorboard/_embedding.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/tensorboard/_embedding.py_docs.md`
- **Size**: 5,789 bytes (5.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/tensorboard/_embedding.py`

## File Metadata

- **Path**: `torch/utils/tensorboard/_embedding.py`
- **Size**: 3,281 bytes (3.20 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import math
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo


_HAS_GFILE_JOIN = hasattr(tf.io.gfile, "join")


def _gfile_join(a, b):
    # The join API is different between tensorboard's TF stub and TF:
    # https://github.com/tensorflow/tensorboard/issues/6080
    # We need to try both because `tf` may point to either the stub or the real TF.
    if _HAS_GFILE_JOIN:
        return tf.io.gfile.join(a, b)
    else:
        fs = tf.io.gfile.get_filesystem(a)
        return fs.join(a, b)


def make_tsv(metadata, save_path, metadata_header=None) -> None:
    if not metadata_header:
        metadata = [str(x) for x in metadata]
    else:
        if len(metadata_header) != len(
            metadata[0]
        ):
            raise AssertionError("len of header must be equal to the number of columns in metadata")
        metadata = ["\t".join(str(e) for e in l) for l in [metadata_header] + metadata]

    metadata_bytes = tf.compat.as_bytes("\n".join(metadata) + "\n")
    with tf.io.gfile.GFile(_gfile_join(save_path, "metadata.tsv"), "wb") as f:
        f.write(metadata_bytes)


# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path) -> None:
    from PIL import Image
    from io import BytesIO

    # this ensures the sprite image has correct dimension as described in
    # https://www.tensorflow.org/get_started/embedding_viz
    nrow = math.ceil((label_img.size(0)) ** 0.5)
    arranged_img_CHW = make_grid(make_np(label_img), ncols=nrow)

    # augment images so that #images equals nrow*nrow
    arranged_augment_square_HWC = np.zeros(
        (arranged_img_CHW.shape[2], arranged_img_CHW.shape[2], 3)
    )
    arranged_img_HWC = arranged_img_CHW.transpose(1, 2, 0)  # chw -> hwc
    arranged_augment_square_HWC[: arranged_img_HWC.shape[0], :, :] = arranged_img_HWC
    im = Image.fromarray(np.uint8((arranged_augment_square_HWC * 255).clip(0, 255)))

    with BytesIO() as buf:
        im.save(buf, format="PNG")
        im_bytes = buf.getvalue()

    with tf.io.gfile.GFile(_gfile_join(save_path, "sprite.png"), "wb") as f:
        f.write(im_bytes)


def get_embedding_info(metadata, label_img, subdir, global_step, tag):
    info = EmbeddingInfo()
    info.tensor_name = f"{tag}:{str(global_step).zfill(5)}"
    info.tensor_path = _gfile_join(subdir, "tensors.tsv")
    if metadata is not None:
        info.metadata_path = _gfile_join(subdir, "metadata.tsv")
    if label_img is not None:
        info.sprite.image_path = _gfile_join(subdir, "sprite.png")
        info.sprite.single_image_dim.extend([label_img.size(3), label_img.size(2)])
    return info


def write_pbtxt(save_path, contents) -> None:
    config_path = _gfile_join(save_path, "projector_config.pbtxt")
    with tf.io.gfile.GFile(config_path, "wb") as f:
        f.write(tf.compat.as_bytes(contents))


def make_mat(matlist, save_path) -> None:
    with tf.io.gfile.GFile(_gfile_join(save_path, "tensors.tsv"), "wb") as f:
        for x in matlist:
            x = [str(i.item()) for i in x]
            f.write(tf.compat.as_bytes("\t".join(x) + "\n"))

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_gfile_join`, `make_tsv`, `make_sprite`, `get_embedding_info`, `write_pbtxt`, `make_mat`

**Key imports**: math, numpy as np, make_np, make_grid, tf, EmbeddingInfo, Image, BytesIO


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/tensorboard`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `numpy as np`
- `._convert_np`: make_np
- `._utils`: make_grid
- `tensorboard.compat`: tf
- `tensorboard.plugins.projector.projector_config_pb2`: EmbeddingInfo
- `PIL`: Image
- `io`: BytesIO


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

Files in the same folder (`torch/utils/tensorboard`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`summary.py_docs.md`](./summary.py_docs.md)
- [`_pytorch_graph.py_docs.md`](./_pytorch_graph.py_docs.md)
- [`_onnx_graph.py_docs.md`](./_onnx_graph.py_docs.md)
- [`_convert_np.py_docs.md`](./_convert_np.py_docs.md)
- [`_proto_graph.py_docs.md`](./_proto_graph.py_docs.md)
- [`writer.py_docs.md`](./writer.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)


## Cross-References

- **File Documentation**: `_embedding.py_docs.md`
- **Keyword Index**: `_embedding.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/tensorboard`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/tensorboard`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/utils/tensorboard`):

- [`writer.py_docs.md_docs.md`](./writer.py_docs.md_docs.md)
- [`_pytorch_graph.py_docs.md_docs.md`](./_pytorch_graph.py_docs.md_docs.md)
- [`_pytorch_graph.py_kw.md_docs.md`](./_pytorch_graph.py_kw.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`_convert_np.py_kw.md_docs.md`](./_convert_np.py_kw.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`_onnx_graph.py_docs.md_docs.md`](./_onnx_graph.py_docs.md_docs.md)
- [`summary.py_docs.md_docs.md`](./summary.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_embedding.py_docs.md_docs.md`
- **Keyword Index**: `_embedding.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
