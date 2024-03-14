
import os
import torch
import numpy as np

from lietorch import SE3

from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.dataloaders.utils.FolderTree import FolderTree
from efm_datasets.dataloaders.utils.misc import update_dict, invert_pose
from efm_datasets.utils.read import read_image, read_numpy


def load_intrinsics(filename):
    """Get hard-coded intrinsics."""
    return np.array([[320.0,   0.0, 320.0],
                     [  0.0, 320.0, 240.0],
                     [  0.0,   0.0,   1.0]])


def load_depth(filename):
    """Get depth from filename."""
    depth = read_numpy(filename.replace('image', 'depth').replace('.png', '_depth.npy'))
    depth[depth == np.nan] = 0.0
    depth[depth == np.inf] = 0.0
    return depth


def load_pose(filename, loc, key, stride):
    """Get pose from filename."""
    # Parses filename to find the pose file
    pose_filename = '/'.join(filename.split('/')[:-2]) + '/pose_%s.txt' % ('left' if 'left' in filename else 'right')
    # Loads pose from a filename
    poses = np.loadtxt(os.path.join(pose_filename), delimiter=' ')
    # Restructures pose to the proper order
    poses = torch.tensor(poses[:, [1, 2, 0, 4, 5, 3, 6]])
    # Return pose relative to the target frame
    return invert_pose(SE3(poses[(loc + key[0]) * stride]).matrix().numpy())


class TartanAirDataset(BaseDataset):
    """TartanAir dataset class. 

    https://theairlab.org/tartanair-dataset/

    Parameters
    ----------
    stride : int, optional
        Temporal context stride, by default None
    """
    def __init__(self, stride=None, **kwargs):
        super().__init__(**kwargs, base_tag='tartan_air')

        remove = {
            'all': None,
            'train': ['soulcity', 'westerndesert'],
        }[self.split]

        self.rgb_tree = FolderTree(
            path=self.path,
            context=self.context, sub_folders=['image_left'], stride=stride, deep=3,
            remove_files=remove, single_folder=False, suffix='.png')
        self.stride = stride

    def __len__(self):
        """Dataset length."""
        return len(self.rgb_tree)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""

        # Initialize sample
        sample, idx = self.initialize_sample(idx)

        for cam_idx, cam in enumerate(self.cameras):

            # Filename
            filename, loc = self.rgb_tree.get_item(idx, return_loc=True)
            if cam == 1:
                filename = {key: val.replace('left', 'right') for key, val in filename.items()}

            for time, val in filename.items():
                time_cam = (time, cam_idx)
                update_dict(sample, 'filename', time_cam, filename)
                update_dict(sample, 'rgb', time_cam, read_image(val))
                update_dict(sample, 'intrinsics', time_cam, load_intrinsics(val))
                if self.with_pose:
                    update_dict(sample, 'pose', time_cam, load_pose(val, loc, time_cam, self.stride))
                if self.with_depth:
                    update_dict(sample, 'depth', time_cam, load_depth(val))

            # If with context
            if self.with_context:
                filename_context = self.rgb_tree.get_context(idx)
                if cam == 1:
                    filename_context = {key: val.replace('left', 'right') for key, val in filename_context.items()}
                for time, val in filename_context.items():
                    time_cam = (time, cam_idx)
                    update_dict(sample, 'filename', time_cam, filename)
                    update_dict(sample, 'rgb', time_cam, read_image(val))
                    update_dict(sample, 'intrinsics', time_cam, load_intrinsics(val))
                    if self.with_pose:
                        update_dict(sample, 'pose', time_cam, load_pose(val, loc, time_cam, self.stride))
                    if self.with_depth_context:
                        update_dict(sample, 'depth', time_cam, load_depth(val))

        # Return post-processed sample
        return self.post_process_sample(sample)

