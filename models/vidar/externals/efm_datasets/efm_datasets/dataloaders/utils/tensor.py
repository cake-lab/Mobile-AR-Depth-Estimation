import torch
import torchvision.transforms as transforms

from efm_datasets.utils.decorators import iterate1


@iterate1
def to_tensor(matrix, tensor_type='torch.FloatTensor'):
    """Casts a matrix to a torch.Tensor"""
    return torch.Tensor(matrix).type(tensor_type)


@iterate1
@iterate1
def to_tensor_permute(matrix, tensor_type='torch.FloatTensor'):
    """Casts a matrix to a torch.Tensor"""
    return torch.Tensor(matrix).type(tensor_type).permute(2, 0, 1)

@iterate1
def to_tensor_image(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)


def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    # Convert using torchvision
    keys = ['rgb', 'mask', 'depth']
    for key_sample, val_sample in sample.items():
        for key in keys:
            if key in key_sample:
                sample[key_sample] = to_tensor_image(val_sample, tensor_type)
    # Convert from numpy
    keys = ['intrinsics', 'pose']
    for key_sample, val_sample in sample.items():
        for key in keys:
            if key in key_sample:
                sample[key_sample] = to_tensor(val_sample, tensor_type)
    # Convert from numpy
    keys = ['optical_flow']
    for key_sample, val_sample in sample.items():
        for key in keys:
            if key in key_sample:
                sample[key_sample] = to_tensor_permute(val_sample, tensor_type)
    # Return converted sample
    return sample
