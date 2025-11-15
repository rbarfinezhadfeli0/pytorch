# Documentation: `docs/docs/source/tensorboard.rst_docs.md`

## File Metadata

- **Path**: `docs/docs/source/tensorboard.rst_docs.md`
- **Size**: 5,647 bytes (5.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/tensorboard.rst`

## File Metadata

- **Path**: `docs/source/tensorboard.rst`
- **Size**: 3,174 bytes (3.10 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
torch.utils.tensorboard
===================================
.. automodule:: torch.utils.tensorboard

Before going further, more details on TensorBoard can be found at
https://www.tensorflow.org/tensorboard/

Once you've installed TensorBoard, these utilities let you log PyTorch models
and metrics into a directory for visualization within the TensorBoard UI.
Scalars, images, histograms, graphs, and embedding visualizations are all
supported for PyTorch models and tensors as well as Caffe2 nets and blobs.

The SummaryWriter class is your main entry to log data for consumption
and visualization by TensorBoard. For example:

.. code:: python


    import torch
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets, transforms

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.close()

This can then be visualized with TensorBoard, which should be installable
and runnable with::

    pip install tensorboard
    tensorboard --logdir=runs


Lots of information can be logged for one experiment. To avoid cluttering
the UI and have better result clustering, we can group plots by naming them
hierarchically. For example, "Loss/train" and "Loss/test" will be grouped
together, while "Accuracy/train" and "Accuracy/test" will be grouped separately
in the TensorBoard interface.

.. code:: python


    from torch.utils.tensorboard import SummaryWriter
    import numpy as np

    writer = SummaryWriter()

    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


Expected result:

.. image:: _static/img/tensorboard/hier_tags.png
    :scale: 75 %

|
|

.. currentmodule:: torch.utils.tensorboard.writer

.. autoclass:: SummaryWriter

   .. automethod:: __init__
   .. automethod:: add_scalar
   .. automethod:: add_scalars
   .. automethod:: add_histogram
   .. automethod:: add_image
   .. automethod:: add_images
   .. automethod:: add_figure
   .. automethod:: add_video
   .. automethod:: add_audio
   .. automethod:: add_text
   .. automethod:: add_graph
   .. automethod:: add_embedding
   .. automethod:: add_pr_curve
   .. automethod:: add_custom_scalars
   .. automethod:: add_mesh
   .. automethod:: add_hparams
   .. automethod:: flush
   .. automethod:: close

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/source`):

- [`torch.compiler_troubleshooting.md_docs.md`](./torch.compiler_troubleshooting.md_docs.md)
- [`torch.compiler_aot_inductor_debugging_guide.md_docs.md`](./torch.compiler_aot_inductor_debugging_guide.md_docs.md)
- [`mtia.memory.md_docs.md`](./mtia.memory.md_docs.md)
- [`torch.compiler_get_started.md_docs.md`](./torch.compiler_get_started.md_docs.md)
- [`torch.compiler_dynamo_deepdive.md_docs.md`](./torch.compiler_dynamo_deepdive.md_docs.md)
- [`mtia.mtia_graph.md_docs.md`](./mtia.mtia_graph.md_docs.md)
- [`hub.md_docs.md`](./hub.md_docs.md)
- [`torch_nccl_environment_variables.md_docs.md`](./torch_nccl_environment_variables.md_docs.md)
- [`optim.md_docs.md`](./optim.md_docs.md)
- [`torch.compiler_aot_inductor.md_docs.md`](./torch.compiler_aot_inductor.md_docs.md)


## Cross-References

- **File Documentation**: `tensorboard.rst_docs.md`
- **Keyword Index**: `tensorboard.rst_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/docs/source`):

- [`distributions.md_docs.md_docs.md`](./distributions.md_docs.md_docs.md)
- [`distributed.optim.md_docs.md_docs.md`](./distributed.optim.md_docs.md_docs.md)
- [`torch.compiler_dynamic_shapes.md_kw.md_docs.md`](./torch.compiler_dynamic_shapes.md_kw.md_docs.md)
- [`tensor_attributes.rst_docs.md_docs.md`](./tensor_attributes.rst_docs.md_docs.md)
- [`tensor_attributes.rst_kw.md_docs.md`](./tensor_attributes.rst_kw.md_docs.md)
- [`torch.compiler_dynamo_overview.md_docs.md_docs.md`](./torch.compiler_dynamo_overview.md_docs.md_docs.md)
- [`mtia.memory.md_kw.md_docs.md`](./mtia.memory.md_kw.md_docs.md)
- [`nn.attention.varlen.md_kw.md_docs.md`](./nn.attention.varlen.md_kw.md_docs.md)
- [`cpu.rst_kw.md_docs.md`](./cpu.rst_kw.md_docs.md)
- [`torch.compiler_faq.md_docs.md_docs.md`](./torch.compiler_faq.md_docs.md_docs.md)


## Cross-References

- **File Documentation**: `tensorboard.rst_docs.md_docs.md`
- **Keyword Index**: `tensorboard.rst_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
