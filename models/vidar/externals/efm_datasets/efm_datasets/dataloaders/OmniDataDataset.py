
import json
import os
import torch
import math
import pytorch3d

import numpy as np
from PIL import Image
from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.dataloaders.utils.FolderTree import FolderTree
from efm_datasets.dataloaders.utils.misc import update_dict
from efm_datasets.utils.read import read_image, read_depth, read_pickle, read_numpy
from efm_datasets.utils.write import write_pickle
from pytorch3d.renderer import FoVPerspectiveCameras
from efm_datasets.dataloaders.utils.misc import invert_pose
from efm_datasets.utils.write import write_empty_txt
from efm_datasets.utils.data import shuffle_dict


class OmniDataDataset(BaseDataset):
    """Omnidata dataset class. 

    https://github.com/EPFL-VILAB/omnidata

    Parameters
    ----------
    filter_invalids : bool, optional
        Remove invalid samples, by default True
    masked_depth : bool, optional
        True if invalid masks are used to mask depth maps, by default True
    """
    def __init__(self, filter_invalids=True, masked_depth=True, **kwargs):
        super().__init__(**kwargs, base_tag='omnidata')
        self.rgb_tree = FolderTree(
            self.path + '/rgb/' + self.split, 
            invalids_end=self.get_invalids(filter_invalids),
            context=self.context, context_type=self.context_type,
            single_folder=False, suffix='.png')
        self.rot_roll = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.masked_depth = masked_depth

    def get_invalids(self, filter_invalids):
        """Return invalid samples."""
        if not filter_invalids:
            return None
        invalid_file = f'{self.path}/invalids/invalids_{self.split}.txt'
        if os.path.exists(invalid_file):
            with open(invalid_file, 'r') as f:
                invalids = f.read().splitlines()
        else:
            invalids = None
        return invalids

    def __len__(self):
        """Dataset length"""
        return len(self.rgb_tree)

    def get_point_file(self, filename):
        """Get point files from filename."""
        if self.split in ['taskonomy']:
            filename = filename.replace('rgb', 'point_info').replace('.png', '.json')
        elif self.split in ['hm3d', 'replica', 'replica_gso']:
            filename = filename.replace('rgb', 'point_info').replace('.png', '.json')
            filename = filename.replace('point_info.json', 'fixatedpose.json')
        else:
            raise ValueError('Invalid split')
        return filename

    def get_rgb(self, filename):
        """Get image from filename."""
        return read_image(filename)

    def get_depth(self, filename):
        """Get depth from filename."""
        depth = read_depth(filename.replace('rgb', 'depth_zbuffer'), div=512)
        if not self.masked_depth:
            return depth
        if self.split in ['taskonomy']:
            mask = read_image(filename.replace('/rgb/', '/mask_valid/').replace('rgb.', 'depth_zbuffer.'), mode='1')
        elif self.split in ['hm3d', 'replica', 'replica_gso']:
            mask = read_image(filename.replace('rgb', 'mask_valid'), mode='1')
        return depth * np.array(mask)

    @staticmethod
    def get_intrinsics_from_data(width, height, fov):
        """Get intrinsics from data"""
        znear, zfar = 0.001, 512.0
        Kfov = FoVPerspectiveCameras(
            device='cpu', fov=fov, degrees=False, R=None, T=None,
        ).compute_projection_matrix(
            znear=znear, zfar=zfar, fov=fov, aspect_ratio=1.0, degrees=False
        )[0]

        K = np.eye(3)
        K[0, 0] = Kfov[0, 0] * width / 2
        K[1, 1] = Kfov[1, 1] * height / 2
        K[0, 2] = width / 2
        K[1, 2] = height / 2

        return K

    def get_intrinsics(self, filename):
        """Get intrinsics from filename."""
        try:
            filename = self.get_point_file(filename)
            data = json.load(open(filename, 'r'))
        except:
            write_empty_txt(filename, 'invalids')
            return None

        width = height = data['resolution']
        fov = data['field_of_view_rads']
        return self.get_intrinsics_from_data(width, height, fov)

    def get_pose(self, filename):
        """Get pose from filename."""
        try:
            filename = self.get_point_file(filename)
            data = json.load(open(filename, 'r'))
        except:
            write_empty_txt(filename, 'invalids')
            return None

        location = data['camera_location']
        rotation = data['camera_rotation_final']

        Tx, Ty, Tz = location
        ex, ey, ez = rotation

        EULER_X_OFFSET_RADS = math.radians(90.0)
        R_inv = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor(
            [((ex - EULER_X_OFFSET_RADS), -ey, -ez)], dtype=torch.double, device='cpu'), 'XZY')
        t_inv = torch.tensor([[-Tx, Tz, Ty]], dtype=torch.double, device='cpu')

        pose = np.eye(4)
        pose[:3, :3] = R_inv.transpose(1,2)
        pose[-1, :3] = - R_inv.bmm(t_inv.unsqueeze(-1)).squeeze(-1)
        pose = np.transpose(pose)
        pose = invert_pose(pose)
        pose[:3, :3] = pose[:3, :3] @ self.rot_roll
        pose = invert_pose(pose)

        return pose
    
    def get_target(self, sample, filename, time_cam):
        """Add target information to sample."""
        update_dict(sample, 'filename', time_cam, filename)
        if self.with_rgb:
            update_dict(sample, 'rgb', time_cam, 
                        self.get_rgb(filename))
        if self.with_intrinsics:
            update_dict(sample, 'intrinsics', time_cam,
                        self.get_intrinsics(filename))
        if self.with_depth:
            update_dict(sample, 'depth', time_cam,
                        self.get_depth(filename))
        if self.with_pose:
            update_dict(sample, 'pose', time_cam,
                            self.get_pose(filename))
        return sample

    def get_context(self, sample, filename_context):
        """Add context information to sample."""
        for time_cam, filename in filename_context.items():
            update_dict(sample, 'filename', time_cam, filename)
            if self.with_rgb_context:
                update_dict(sample, 'rgb', time_cam,
                            self.get_rgb(filename))
            if self.with_intrinsics_context:
                update_dict(sample, 'intrinsics', time_cam,
                            self.get_intrinsics(filename))
            if self.with_depth_context:
                update_dict(sample, 'depth', time_cam,
                            self.get_depth(filename))
            if self.with_pose_context:
                update_dict(sample, 'pose', time_cam,
                            self.get_pose(filename))
        return sample

    def get_filename_target(self, idx, cam):
        """Get target filename."""
        return self.rgb_tree.get_item(idx)[0]

    def get_filename_context(self, idx, cam_idx, cam):
        """Get context filename."""
        if 'sequence' in self.context_type:
            if self.temporal_proximity is None:
                filename_context = self.get_sequence_context(idx, cam)
            else:
                filename = self.get_filename_target(idx, cam)
                filename_context = self.get_temporal_proximity(idx, cam, filename)
        else:
            filename_context = self.rgb_tree.get_context(idx)
        if self.context_sample is not None:
            filename_context = shuffle_dict(filename_context, self.context_sample)
        return {(key, cam_idx): val for key, val in filename_context.items()}

    def __getitem__(self, idx):
        """Get dataset sample given an index"""

        # Initialize sample
        sample, idx = self.initialize_sample(idx)
        cameras = self.get_sample_cameras(idx)

        # Loop over all requested cameras
        for cam_idx, cam in enumerate(cameras):
            filename = self.get_filename_target(idx, cam)
            sample = self.get_target(sample, filename, (0, cam_idx))
            if self.with_context:
                filename_context = self.get_filename_context(idx, cam_idx, cam)
                sample = self.get_context(sample, filename_context)

        # Return sample
        return self.post_process_sample(sample)
