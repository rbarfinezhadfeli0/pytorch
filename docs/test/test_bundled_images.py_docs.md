# Documentation: `test/test_bundled_images.py`

## File Metadata

- **Path**: `test/test_bundled_images.py`
- **Size**: 3,310 bytes (3.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]
# mypy: allow-untyped-defs

import io

import cv2  # @manual

import torch
import torch.utils.bundled_inputs
from torch.testing._internal.common_utils import TestCase


torch.ops.load_library("//caffe2/torch/fb/operators:decode_bundled_image")


def model_size(sm):
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    return len(buffer.getvalue())


def save_and_load(sm):
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)


"""Return an InflatableArg that contains a tensor of the compressed image and the way to decode it

    keyword arguments:
    img_tensor -- the raw image tensor in HWC or NCHW with pixel value of type unsigned int
                  if in NCHW format, N should be 1
    quality -- the quality needed to compress the image
"""


def bundle_jpeg_image(img_tensor, quality):
    # turn NCHW to HWC
    if img_tensor.dim() == 4:
        assert img_tensor.size(0) == 1
        img_tensor = img_tensor[0].permute(1, 2, 0)
    pixels = img_tensor.numpy()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc_img = cv2.imencode(".JPEG", pixels, encode_param)
    enc_img_tensor = torch.from_numpy(enc_img)
    enc_img_tensor = torch.flatten(enc_img_tensor).byte()
    obj = torch.utils.bundled_inputs.InflatableArg(
        enc_img_tensor, "torch.ops.fb.decode_bundled_image({})"
    )
    return obj


def get_tensor_from_raw_BGR(im) -> torch.Tensor:
    raw_data = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    raw_data = torch.from_numpy(raw_data).float()
    raw_data = raw_data.permute(2, 0, 1)
    raw_data = torch.div(raw_data, 255).unsqueeze(0)
    return raw_data


class TestBundledImages(TestCase):
    def test_single_tensors(self):
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg

        im = cv2.imread("caffe2/test/test_img/p1.jpg")
        tensor = torch.from_numpy(im)
        inflatable_arg = bundle_jpeg_image(tensor, 90)
        input = [(inflatable_arg,)]
        sm = torch.jit.script(SingleTensorModel())
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(sm, input)
        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        decoded_data = inflated[0][0]

        # raw image
        raw_data = get_tensor_from_raw_BGR(im)

        self.assertEqual(len(inflated), 1)
        self.assertEqual(len(inflated[0]), 1)
        self.assertEqual(raw_data.shape, decoded_data.shape)
        self.assertEqual(raw_data, decoded_data, atol=0.1, rtol=1e-01)

        # Check if fb::image_decode_to_NCHW works as expected
        with open("caffe2/test/test_img/p1.jpg", "rb") as fp:
            weight = torch.full((3,), 1.0 / 255.0).diag()
            bias = torch.zeros(3)
            byte_tensor = torch.tensor(list(fp.read())).byte()
            im2_tensor = torch.ops.fb.image_decode_to_NCHW(byte_tensor, weight, bias)
            self.assertEqual(raw_data.shape, im2_tensor.shape)
            self.assertEqual(raw_data, im2_tensor, atol=0.1, rtol=1e-01)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview

"""Return an InflatableArg that contains a tensor of the compressed image and the way to decode it    keyword arguments:    img_tensor -- the raw image tensor in HWC or NCHW with pixel value of type unsigned int                  if in NCHW format, N should be 1    quality -- the quality needed to compress the image

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestBundledImages`, `SingleTensorModel`

**Functions defined**: `model_size`, `save_and_load`, `bundle_jpeg_image`, `get_tensor_from_raw_BGR`, `test_single_tensors`, `forward`

**Key imports**: io, cv2  , torch, torch.utils.bundled_inputs, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `cv2  `
- `torch`
- `torch.utils.bundled_inputs`
- `torch.testing._internal.common_utils`: TestCase


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/test_bundled_images.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_bundled_images.py_docs.md`
- **Keyword Index**: `test_bundled_images.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
