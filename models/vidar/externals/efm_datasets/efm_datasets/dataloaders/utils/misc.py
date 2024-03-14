import random

import numpy as np
import torch

from efm_datasets.utils.decorators import iterate1
from efm_datasets.utils.types import is_seq, is_tensor, is_dict, is_int


def update_dict(data, key1, key2, val):
    """Update dictionary with a new key-value pair"""
    if val is None:
        return
    if key1 not in data:
        data[key1] = {}
    if key2 not in data[key1]:
        data[key1][key2] = {}
    data[key1][key2] = val


def update_dict_nested(data, key1, key2, key3, val):
    """Update dictionary with multiple key-value pairs"""
    if val is None:
        return
    if key1 not in data:
        data[key1] = {}
    if key2 not in data[key1]:
        data[key1][key2] = {}
    if key3 not in data[key1][key2]:
        data[key1][key2][key3] = {}
    data[key1][key2][key3] = val


def stack_sample(sample, lidar_sample=None, radar_sample=None):
    """Stack sample into a single dictionary"""
    # If there are no tensors, return empty list
    if len(sample) == 0:
        return None
    # If there is only one sensor don't do anything
    if len(sample) == 1:
        sample = sample[0]
        return sample
    # Otherwise, stack sample
    first_sample = sample[0]
    stacked_sample = {}
    for key, val in first_sample.items():
        # Global keys (do not stack)
        if key in ['idx', 'dataset_idx']:
            stacked_sample[key] = first_sample[key]
        # Meta keys
        elif key in ['meta']:
            stacked_sample[key] = {}
            for key2 in first_sample[key].keys():
                stacked_sample[key][key2] = {}
                for key3 in first_sample[key][key2].keys():
                    stacked_sample[key][key2][key3] = torch.stack(
                        [torch.tensor(s[key][key2][key3]) for s in sample], 0)
        # Stack tensors
        elif is_tensor(val):
            stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
        # Stack list
        elif is_seq(first_sample[key]):
            stacked_sample[key] = []
            # Stack list of torch tensors
            if is_tensor(first_sample[key][0]):
                for i in range(len(first_sample[key])):
                    stacked_sample[key].append(
                        torch.stack([s[key][i] for s in sample], 0))
            else:
                stacked_sample[key] = [s[key] for s in sample]
        # Repeat for dictionaries
        elif is_dict(first_sample[key]):
            stacked_sample[key] = stack_sample([s[key] for s in sample])
        # Append lists
        else:
            stacked_sample[key] = [s[key] for s in sample]

    # Return stacked sample
    return stacked_sample


@iterate1
def invert_pose(pose):
    """
    Inverts a transformation matrix (pose)

    Parameters
    ----------
    pose : np.array
        Input pose [4, 4]

    Returns
    -------
    inv_pose : np.array
        Inverted pose [4, 4]
    """
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = np.transpose(pose[:3, :3])
    inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
    return inv_pose


def make_relative_pose(sample, tgt=(0, 0)):
    """Make sample poses relative to a target camera"""
    # Do nothing if there is no pose
    if 'pose' not in sample:
        return sample
    pose = sample['pose']
    # Get inverse current poses
    inv_pose = {key: invert_pose(val) for key, val in pose.items() if key[0] == tgt[0]}
    # For each camera
    for key, val in pose.items():
        if key[0] == tgt[0]:
            if key[1] != tgt[1]:
                pose[key] = pose[key] @ inv_pose[tgt]
        else:
            pose[key] = pose[key] @ inv_pose[(tgt[0], key[1])]
    return sample


def dummy_intrinsics(image):
    """
    Return dummy intrinsics calculated based on image resolution

    Parameters
    ----------
    image : PIL.Image
        Image from which intrinsics will be calculated

    Returns
    -------
    intrinsics : np.array (3x3)
        Image intrinsics (fx = cx = w/2, fy = cy = h/2)
    """
    w, h = [float(d) for d in image.size]
    return np.array([[w/2, 0., w/2. - 0.5],
                     [0., h/2, h/2. - 0.5],
                     [0., 0., 1.]])

