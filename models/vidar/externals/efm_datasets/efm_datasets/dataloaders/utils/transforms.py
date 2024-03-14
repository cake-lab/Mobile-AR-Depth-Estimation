
from efm_datasets.dataloaders.utils.tensor import to_tensor_sample


def no_transform(sample):
    sample = to_tensor_sample(sample)
    return sample
