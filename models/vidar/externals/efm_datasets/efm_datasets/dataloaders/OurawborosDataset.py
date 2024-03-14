
import os
import json
import numpy as np

from PIL import Image

from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.dataloaders.utils.FolderTree import FolderTree
from efm_datasets.dataloaders.utils.misc import update_dict, update_dict_nested, invert_pose
from efm_datasets.utils.read import read_image, read_numpy
from efm_datasets.utils.types import is_str
from efm_datasets.utils.data import shuffle_dict


class OurawborosDataset(BaseDataset):
    """Ourawboros dataset class.  Used to read preprocessed Ouroboros data.

    Parameters
    ----------
    depth_type : str, optional
        Depth map type, by default None
    prefix : str, optional
        Camera prefix, by default 'camera'
    resolution : _type_, optional
        Requested sample resolution, by default None
    rgb_extension : _type_, optional
        Image extension, by default None
    filter_invalids : bool, optional
        Remove invalid samples, by default True
    zero_origin : bool, optional
        True if poses are relative to the beginning of the sequence, by default False
    base_camera : int, optional
        Base camera used as origin, by default None
    num_images : int, optional
        How many images should be considered (useful for debugging and overfitting), by default None
    """
    def __init__(self, depth_type=None, prefix='camera', resolution=None, rgb_extension=None, filter_invalids=True,
                 zero_origin=False, base_camera=None, num_images=None, **kwargs):
        super().__init__(**kwargs, base_tag='ourawboros')

        self.prefix = prefix
        self.depth_type = depth_type
        self.base_camera = self.cameras[0] if base_camera is None else base_camera

        resolution = '' if resolution is None else '_%d_%d' % tuple(resolution)

        self.rgb_folder = f'rgb{resolution}'
        self.intrinsics_folder = f'intrinsics{resolution}'
        self.extrinsics_folder = 'extrinsics'
        self.pose_folder = 'pose'
        self.semantic_folder = 'semantic_segmentation_2d'
        self.depth_folder = \
            'depth' if resolution == '' and depth_type == 'zbuffer' else \
                f'projected/depth{resolution}/{depth_type}'
        self.bwd_optflow_folder = 'back_motion_vectors_2d' if resolution == '' else \
            f'bwd_optical_flow{resolution}'
        self.fwd_optflow_folder = 'motion_vectors_2d' if resolution == '' else \
            f'fwd_optical_flow{resolution}'

        self.rgb_extension = rgb_extension if rgb_extension is not None else '.png' if resolution == '' else '.jpg'
        self.optflow_extension = '.png' if resolution == '' else '.npz'
        self.depth_key = 'data' if resolution == '' and depth_type == 'zbuffer' else 'depth'

        cameras = list(set(self.cameras + [self.base_camera] + self.cameras_context))
        # Store variables
        if self.path.endswith('.json'):
            self.path, self.json = os.path.dirname(self.path), os.path.basename(self.path)
            # Create data tree
            self.rgb_tree = {key: FolderTree(
                self.path, context=self.context,
                context_type=self.context_type, sub_folders=[self.get_camera(key)],
                nested=True, filter_nested=self.rgb_folder,
                keep_folders=self.get_split(self.split, filter_invalids),
                single_folder=False, suffix=self.rgb_extension, finish=num_images)
                for key in cameras}
        else:
            self.rgb_tree = {key: FolderTree(
                self.path, context=self.context,
                context_type=self.context_type, sub_folders=[self.get_camera(key)],
                nested=True, filter_nested=self.rgb_folder,
                single_folder=False, suffix='.jpg', finish=num_images)
                for key in cameras}

        # If zero_origin is requested, the first sample of each scene is set to identity
        if zero_origin:
            sample = self.__getitem__(0, force_camera=self.base_camera)
            self.base_pose = invert_pose(sample['pose'][(0, 0)].cpu().numpy())

    def __len__(self):
        """Dataset length"""
        return len(self.rgb_tree[self.base_camera]) if self.fixed_idx is None else 1

    def get_camera(self, key):
        """Parse camera key to get relevant camera folder"""
        return key if is_str(key) else f'{self.prefix}_%02d' % key

    def get_split(self, split, filter_invalids):
        """Parse split to get relevant scenes"""
        split = {'train': '0', 'val': '1'}[split]
        json_file = os.path.join(self.path, self.json)
        with open(json_file, "r") as read_content:
            data = json.load(read_content)
        data = data['scene_splits'][split]['filenames']
        data = [d.split('/')[0] for d in data]
        if filter_invalids:
            invalid_file = os.path.join(self.path, self.json.replace('.json', '_invalids.txt'))
            if os.path.exists(invalid_file):
                with open(invalid_file, "r") as read_content:
                    invalid = read_content.read().split('\n')
                    invalid = [val for val in invalid if len(val) > 0]
                    invalid = [val[:-4] if val.endswith('.txt') else val for val in invalid]
                    data = [d for d in data if d not in invalid]
        return data

    def get_rgb(self, filename):
        """Get image from filename."""
        return read_image(filename)

    def get_intrinsics(self, filename):
        """Get intrinsics from filename."""
        filename = filename.replace(self.rgb_folder, self.intrinsics_folder)[:-4] + '.npy'
        return read_numpy(filename)

    def get_pose(self, filename):
        """Get pose from filename."""
        filename = filename.replace(self.rgb_folder, self.pose_folder)[:-4] + '.npy'
        return read_numpy(filename)

    def get_extrinsics(self, filename):
        """Get extrinsics from filename."""
        filename = filename.replace(self.rgb_folder, self.extrinsics_folder)[:-4] + '.npy'
        return read_numpy(filename)

    def get_depth(self, filename):
        """Get depth from filename."""
        filename = filename.replace(self.rgb_folder, self.depth_folder)[:-4] + '.npz'
        return read_numpy(filename, self.depth_key)

    def get_semantic(self, filename):
        """Get semantic from filename."""
        filename = filename.replace(self.rgb_folder, self.semantic_folder)[:-4] + '.png'
        return np.expand_dims(np.array(read_image(filename, mode=''))[..., 0], 0)

    def get_optical_flow(self, filename, direction):
        """Get optical flow from filename."""
        # Check if direction is valid
        optflow_folder = {'fwd': self.fwd_optflow_folder,
                          'bwd': self.bwd_optflow_folder}[direction]
        # Get filename path and load optical flow
        filename = filename.replace(self.rgb_folder, optflow_folder)[:-4] + self.optflow_extension
        if not os.path.exists(filename):
            return None
        elif filename.endswith('.png'):
            optflow = np.array(Image.open(filename))
            # Convert to uv motion
            dx_i = optflow[..., 0] + optflow[..., 1] * 256
            dy_i = optflow[..., 2] + optflow[..., 3] * 256
            dx = ((dx_i / 65535.0) * 2.0 - 1.0) * optflow.shape[1]
            dy = ((dy_i / 65535.0) * 2.0 - 1.0) * optflow.shape[0]
            # Return stacked array
            return np.stack((dx, dy), 2)
        elif filename.endswith('.npz'):
            return np.load(filename)['optflow']
        else:
            raise ValueError('Invalid optical flow extension')
        
    def get_target(self, sample, filename, time_cam):
        """Add target information to sample."""
        update_dict(sample, 'filename', time_cam, filename)
        if self.with_rgb:
            update_dict(sample, 'rgb', time_cam, 
                        self.get_rgb(filename))
        if self.with_intrinsics:
            update_dict(sample, 'intrinsics', time_cam,
                        self.get_intrinsics(filename))
        if self.with_pose:
            update_dict(sample, 'pose', time_cam,
                        self.get_pose(filename))
        if self.with_extrinsics:
            update_dict(sample, 'extrinsics', time_cam,
                        self.get_extrinsics(filename))
        if self.with_depth:
            update_dict(sample, 'depth', time_cam,
                        self.get_depth(filename))
        if self.with_optical_flow:
            if self.within_context(time_cam[0], 'bwd'):
                update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] - 1, time_cam[1]),
                                    self.get_optical_flow(filename, 'bwd'))
            if self.within_context(time_cam[0], 'fwd'):
                update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] + 1, time_cam[1]),
                                    self.get_optical_flow(filename, 'fwd'))
        if self.with_semantic:
            update_dict(sample, 'semantic', time_cam,
                        self.get_semantic(filename))
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
            if self.with_pose_context:
                update_dict(sample, 'pose', time_cam,
                            self.get_pose(filename))
            if self.with_extrinsics_context:
                update_dict(sample, 'extrinsics', time_cam,
                            self.get_extrinsics(filename))
            if self.with_depth_context:
                update_dict(sample, 'depth', time_cam,
                            self.get_depth(filename))
            if self.with_semantic_context:
                update_dict(sample, 'semantic', time_cam,
                            self.get_semantic(filename))
            if self.with_optical_flow_context:
                if self.within_context(time_cam[0], 'bwd'):
                    update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] - 1, time_cam[1]),
                                        self.get_optical_flow(filename, 'bwd'))
                if self.within_context(time_cam[0], 'fwd'):
                    update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] + 1, time_cam[1]),
                                        self.get_optical_flow(filename, 'fwd'))        
        return sample
    
    def get_filename_target(self, idx, cam):
        """Get target filename."""
        return self.rgb_tree[cam].get_item(idx)[0]

    def get_filename_context(self, idx, cam_idx, cam):
        """Get context filename."""
        if 'sequence' in self.context_type:
            if self.temporal_proximity is None:
                filename_context = self.get_sequence_context(idx, cam)
            else:
                filename = self.get_filename_target(idx, cam)
                filename_context = self.get_temporal_proximity(idx, cam, filename)
        else:
            filename_context = self.rgb_tree[cam].get_context(idx)
        if self.context_sample is not None:
            filename_context = shuffle_dict(filename_context, self.context_sample)
        return {(key, cam_idx): val for key, val in filename_context.items()}

    def __getitem__(self, idx, force_camera=None):
        """Get dataset sample given an index."""

        # Initialize sample
        sample, idx = self.initialize_sample(idx)
        cameras = self.get_sample_cameras(idx, force_camera)

        # Loop over all requested cameras
        for cam_idx, cam in enumerate(cameras):
            filename = self.get_filename_target(idx, cam)
            sample = self.get_target(sample, filename, (0, cam_idx))
            if self.with_context:
                filename_context = self.get_filename_context(idx, cam_idx, cam)
                sample = self.get_context(sample, filename_context)

        # Return sample
        return self.post_process_sample(sample)
