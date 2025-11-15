# Documentation: quantization_util.py

## File Metadata
- **Path**: `test/quantization/core/experimental/quantization_util.py`
- **Size**: 5043 bytes
- **Lines**: 149
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
import torch
import torchvision
import torchvision.transforms.transforms as transforms
import os
import torch.ao.quantization
from torchvision.models.quantization.resnet import resnet18
from torch.autograd import Variable

# Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

"""
Define helper functions for APoT PTQ and QAT
"""

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

train_batch_size = 30
eval_batch_size = 50

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)

            loss = criterion(output, target)  # noqa: F841
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print()

    return top1, top5

def load_model(model_file):
    model = resnet18(pretrained=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to("cpu")
    return model

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")

def prepare_data_loaders(data_path):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(data_path,
                                            split="train",
                                            transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                          transforms.RandomHorizontalFlip(),
                                                                          transforms.ToTensor(),
                                                                          normalize]))
    dataset_test = torchvision.datasets.ImageNet(data_path,
                                                 split="val",
                                                 transform=transforms.Compose([transforms.Resize(256),
                                                                               transforms.CenterCrop(224),
                                                                               transforms.ToTensor(),
                                                                               normalize]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

def training_loop(model, criterion, data_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, correct, total = 0, 0, 0
    model.train()
    for _ in range(10):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss = Variable(loss, requires_grad=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return train_loss, correct, total

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): AverageMeter

### Functions
This file defines 10 function(s): __init__, reset, update, __str__, accuracy, evaluate, load_model, print_size_of_model, prepare_data_loaders, training_loop


## Key Components

The file contains 367 words across 149 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5043 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
