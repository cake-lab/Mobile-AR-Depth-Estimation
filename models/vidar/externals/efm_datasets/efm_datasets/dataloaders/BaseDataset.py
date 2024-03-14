import os
from abc import ABC

import torch
import random

from torch.utils.data import Dataset
from efm_datasets.utils.types import is_list, is_str, is_dict
from efm_datasets.dataloaders.utils.misc import make_relative_pose
from efm_datasets.dataloaders.utils.misc import update_dict, update_dict_nested, invert_pose
from efm_datasets.utils.read import read_numpy
import numpy as np


class BaseDataset(Dataset, ABC):
    """Base dataset class, with functionalities shared across all subclasses.

    Parameters
    ----------
    path : str
        Dataset folder
    split : str, optional
        Dataset split (specific to each dataset), by default None
    tag : str, optional
        Dataset tag, to identify its samples in a batch, by default None
    base_tag : str, optional
        Default dataset tag, in the case none is provided, by default None
    make_relative : bool, optional
        True if context poses are returned relative to target pose, by default True
    context : tuple, optional
        Sample temporal context, by default ()
    cameras : tuple, optional
        Sample cameras, by default (0,)
    labels : tuple, optional
        Returned sample target labels, by default ()
    labels_context : tuple, optional
        Returned sample context labels, by default ()
    fixed_idx : int, optional
        True if a single idx is always returned (useful for debugging), by default None
    data_transform : Function, optional
        Transformations used for data augmentation, by default None
    mask_range : tuple, optional
        minimum and maximum depth range (other values are set to zero), by default None
    clip_depth : float, optional
        Maximum depth value (longer values are set to maximum), by default None
    spatial_proximity : list, optional
        Parameters for spatial proximity, by default None
    temporal_proximity : list, optional
        Parameters for temporal proximity, by default None
    cameras_context : tuple, optional
        Sample spatial camera context, by default ()
    context_sample : int, optional
        Randomly subsamples the temporal context to that number, by default None
    cameras_context_sample : int, optional
        Randomly subsamples the spatial context to that number, by default None
    """    
    def __init__(self, path, split=None, tag=None, base_tag=None, make_relative=True,
                 context=(), cameras=(0,), labels=(), labels_context=(), fixed_idx=None,
                 data_transform=None, mask_range=None, clip_depth=None,
                 spatial_proximity=None, temporal_proximity=None, cameras_context=(),
                 context_sample=None, cameras_context_sample=None,
                 **kwargs):
        super().__init__()

        self.path = path
        self.labels = labels
        self.labels_context = labels_context
        self.cameras = self.prepare_cameras(cameras)
        self.data_transform = data_transform
        self.tag = tag if tag is not None else base_tag
        self.make_relative = make_relative
        self.split = split
        self.fixed_idx = fixed_idx

        self.num_cameras = len(cameras) if is_list(cameras) else cameras

        self.with_cameras_context = len(cameras_context) > 0
        if not self.with_cameras_context:
            self.cameras_context = []
        elif len(cameras_context) == 1 and cameras_context[0] == 'proximity':
            self.cameras_context = cameras_context[0]
        else:
            self.cameras_context = self.prepare_cameras(cameras_context)

        self.context_sample = context_sample
        self.cameras_context_sample = cameras_context_sample

        self.context_type = None
        if len(context) == 0 or len(context) > 1 or not is_str(context[0]):
            self.context_type = 'temporal'
        elif len(context) == 1 and is_str(context[0]):
            self.context_type = context[0]
            context = []

        self.bwd_contexts = [ctx for ctx in context if ctx < 0]
        self.fwd_contexts = [ctx for ctx in context if ctx > 0]

        self.bwd_context = 0 if len(context) == 0 else - min(0, min(context))
        self.fwd_context = 0 if len(context) == 0 else max(0, max(context))

        self.context = [v for v in range(- self.bwd_context, 0)] + \
                       [v for v in range(1, self.fwd_context + 1)]

        self.num_context = self.bwd_context + self.fwd_context

        self.with_context = self.context_type != 'temporal' or \
                           (self.context_type == 'temporal' and self.num_context > 0)

        self.base_pose = None

        self.mask_range = mask_range
        self.clip_depth = clip_depth

        self.spatial_proximity = None if spatial_proximity is None else dict(
            sample=spatial_proximity[0],
            min_overlap=spatial_proximity[1],
            min_dist=spatial_proximity[2][0],
            max_dist=spatial_proximity[2][1],
            max_angle=spatial_proximity[3],
        )

        self.temporal_proximity = None if temporal_proximity is None else dict(
            sample=temporal_proximity[0],
            min_overlap=temporal_proximity[1],
            min_dist=temporal_proximity[2][0],
            max_dist=temporal_proximity[2][1],
            max_angle=temporal_proximity[3],
        )

    def prepare_cameras(self, cameras):
        """Parse and return relevant cameras.

        Parameters
        ----------
        cameras : list or string
            Camera information to be parsed

        Returns
        -------
        list
            Relevant cameras
        """
        all_cameras = []
        for camera in cameras:
            if not is_str(camera):
                all_cameras.append(camera)
            elif camera.count('|') != 2:
                all_cameras.append(camera)
            else:
                prefix, st, fn = camera.split('|')
                st, fn = int(st), int(fn)
                for i in range(st, fn):
                    all_cameras.append(f'{prefix}{i}')
        return all_cameras

    def get_sample_cameras(self, idx, force_camera=None):
        """Parse and return sample cameras.

        Parameters
        ----------
        idx : int
            Sample index
        force_camera : int, optional
            Force the sample to return that camera, by default None

        Returns
        -------
        list
            Sample cameras
        """
        if not self.with_cameras_context:
            return self.cameras if force_camera is None else [force_camera]
        elif self.spatial_proximity is not None:
            filename = self.get_filename_target(idx, self.cameras[0])
            cameras = self.get_spatial_proximity(idx, self.cameras[0], filename)
            return self.cameras + cameras
        else:
            cameras = list(self.cameras_context)
            for cam in self.cameras:
                cameras.remove(cam)
            if self.cameras_context_sample is not None:
                random.shuffle(cameras)
                cameras = cameras[:self.cameras_context_sample]
            return self.cameras + cameras

    def calc_proximity(self, proximity, idx, mode):
        """Calculates proximity between cameras

        Parameters
        ----------
        proximity : dict
            Proximity parameters
        idx : int
            Sample index
        mode : str
            Proximity mode [spatial,temporal]

        Returns
        -------
        list
            Nearby cameras given the proximity parametersw
        """

        params = {
            'spatial': self.spatial_proximity,
            'temporal': self.temporal_proximity,
        }[mode]

        max_overlap = proximity[idx, 3]
        min_overlap = params['min_overlap'] * max_overlap

        dist = proximity[:, [1]]
        angle = proximity[:, [2]]
        overlap = proximity[:, [3]]

        valid_dist = ((dist >= params['min_dist']) &
                      (dist <= params['max_dist']))
        valid_angle = angle < params['max_angle']
        valid_overlap = overlap >= min_overlap

        valid = (valid_dist & valid_angle & valid_overlap).squeeze(-1)
        proximity = proximity[valid]

        if params['sample'] != -1:
            rand = torch.randperm(proximity.shape[0])[:params['sample']]
            proximity = proximity[rand]

        return [int(p) for p in proximity[:, 0]]

    def get_sequence_context(self, idx, cam):
        """Get temporal context for a sample

        Parameters
        ----------
        idx : int
            Sample index
        cam : int
            Camera index

        Returns
        -------
        dict
            Context indexes
        """
        rgb_tree = self.rgb_tree[cam] if is_dict(self.rgb_tree) else self.rgb_tree
        idx1, context = rgb_tree.get_context_idxs(idx)
        if self.context_sample is not None:
            random.shuffle(context)
            context = context[:self.context_sample]
        return {
            i+1: rgb_tree.get_proximity(idx1, context[i])
            for i in range(len(context))
        }

    def get_spatial_proximity(self, idx, cam, filename):
        """Get spatial context for a sample 

        Parameters
        ----------
        idx : int
            Sample index
        cam : int
            Camera index
        filename : _type_
            Target filename

        Returns
        -------
        list
            Context filenames
        """
        rgb_tree = self.rgb_tree[cam] if is_dict(self.rgb_tree) else self.rgb_tree
        _, idx2 = rgb_tree.get_idxs(idx)

        proximity_filename = filename.replace('rgb', f'proximity_spatial').replace('.png', '.npy')

        proximity = read_numpy(proximity_filename)
        prox = self.calc_proximity(proximity, idx2, 'spatial')

        return [f'{self.prefix}{prox[i]}' for i in range(len(prox))]

    def get_temporal_proximity(self, idx, cam, filename):
        """Get temporal context for a sample 

        Parameters
        ----------
        idx : int
            Sample index
        cam : int
            Camera index
        filename : _type_
            Target filename

        Returns
        -------
        dict
            Context filenames
        """
        rgb_tree = self.rgb_tree[cam] if is_dict(self.rgb_tree) else self.rgb_tree
        idx1, idx2 = rgb_tree.get_idxs(idx)

        proximity_filename = filename.replace('rgb', f'proximity_temporal')[:-4] + '.npy'

        proximity = read_numpy(proximity_filename)
        prox = self.calc_proximity(proximity, idx2, 'temporal')

        return {
            i + 1: rgb_tree.get_proximity(idx1, prox[i])
            for i in range(len(prox))
        }

    def initialize_sample(self, idx):
        """Initialize sample with basic information

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        dict
            Initialized sample
        int
            Modified sample index
        """
        if self.fixed_idx is not None:
            idx = self.fixed_idx
        return {
            'idx': idx,
            'tag': self.tag,
            'timestep': idx + self.bwd_context,
        }, idx

    def post_process_sample(self, sample):
        """Post-process sample with basic functionality

        Parameters
        ----------
        sample : dict
            Sample to post-process

        Returns
        -------
        dict
            Post-processed sample 
        """
        if self.base_pose is not None:
            for key in sample['pose'].keys():
                sample['pose'][key] = sample['pose'][key] @ self.base_pose
        if self.make_relative:
            sample = make_relative_pose(sample)
        if self.data_transform:
            sample = self.data_transform(sample)
        if self.mask_range is not None:
            for key in ['depth']:
                for tgt in sample[key].keys():
                    invalid = (sample[key][tgt] < self.mask_range[0]) | (sample[key][tgt] > self.mask_range[1])
                    sample[key][tgt][invalid] = 0.0
        if self.clip_depth is not None:
            for key in ['depth']:
                for tgt in sample[key].keys():
                    invalid = sample[key][tgt] > self.clip_depth
                    sample[key][tgt][invalid] = self.clip_depth
        return sample

    def add_data(self, filename, sample, time_cam):
        """Base function to add target labels  to a sample

        Parameters
        ----------
        filename : str
            Sample target filename
        sample : dict
            Sample
        time_cam : tuple
            Timestep and camera indexes

        Returns
        -------
        dict    
            Updated sample
        """
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
        if self.with_depth:
            update_dict(sample, 'depth', time_cam,
                        self.get_depth(filename))
        if self.with_optical_flow:
            update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] - 1, time_cam[1]),
                               self.get_optical_flow(filename, 'bwd'))
            update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] + 1, time_cam[1]),
                               self.get_optical_flow(filename, 'fwd'))
        if self.with_scene_flow:
            update_dict_nested(sample, 'scene_flow', time_cam, (time_cam[0] - 1, time_cam[1]),
                               self.get_scene_flow(filename, 'bwd'))
            update_dict_nested(sample, 'scene_flow', time_cam, (time_cam[0] + 1, time_cam[1]),
                               self.get_scene_flow(filename, 'fwd'))

        return sample

    def add_data_context(self, filename, sample, time_cam):
        """Base function to add context labels  to a sample

        Parameters
        ----------
        filename : str
            Sample context filename
        sample : dict
            Sample
        time_cam : tuple
            Timestep and camera indexes

        Returns
        -------
        dict    
            Updated sample
        """
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
        if self.with_depth_context:
            update_dict(sample, 'depth', time_cam,
                        self.get_depth(filename))
        if self.with_optical_flow_context:
            if self.within_context(time_cam[0], 'bwd'):
                update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] - 1, time_cam[1]),
                                   self.get_optical_flow(filename, 'bwd'))
            if self.within_context(time_cam[0], 'fwd'):
                update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] + 1, time_cam[1]),
                                   self.get_optical_flow(filename, 'fwd'))
        if self.with_scene_flow_context:
            if self.within_context(time_cam[0], 'bwd'):
                update_dict_nested(sample, 'scene_flow', time_cam, (time_cam[0] - 1, time_cam[1]),
                                   self.get_scene_flow(filename, 'bwd'))
            if self.within_context(time_cam[0], 'fwd'):
                update_dict_nested(sample, 'scene_flow', time_cam, (time_cam[0] + 1, time_cam[1]),
                                   self.get_scene_flow(filename, 'fwd'))
        return sample

    def add_dummy_data(self, sample, time_cam):
        """Base function to add dummy target labels to a sample

        Parameters
        ----------
        filename : str
            Sample target filename
        sample : dict
            Sample
        time_cam : tuple
            Timestep and camera indexes

        Returns
        -------
        dict    
            Updated sample
        """
        if self.with_dummy_intrinsics:
            w, h = sample['rgb'][time_cam].size
            update_dict(sample, 'intrinsics', time_cam,
                        np.array([[w / 2, 0, w / 2], [0, h / 2, h / 2], [0, 0, 1]]))
        if self.with_dummy_pose:
            update_dict(sample, 'pose', time_cam, np.eye(4))
        if self.with_dummy_depth:
            update_dict(sample, 'depth', time_cam, np.zeros(sample['rgb'][time_cam].size[::-1]))
        return sample

    def add_dummy_data_context(self, sample, time_cam):
        """Base function to add dummy context labels to a sample

        Parameters
        ----------
        filename : str
            Sample context filename
        sample : dict
            Sample
        time_cam : tuple
            Timestep and camera indexes

        Returns
        -------
        dict    
            Updated sample
        """
        if self.with_dummy_intrinsics_context:
            w, h = sample['rgb'][time_cam].size
            update_dict(sample, 'intrinsics', time_cam, 
                        np.array([[w / 2, 0, w / 2], [0, h / 2, h / 2], [0, 0, 1]]))
        if self.with_dummy_pose_context:
            update_dict(sample, 'pose', time_cam, np.eye(4))
        if self.with_dummy_depth_context:
            update_dict(sample, 'depth', time_cam, np.zeros(sample['rgb'][time_cam].size[::-1]))
        return sample

    def relative_path(self, filename):
        return {key: os.path.splitext(val.replace(self.path + '/', ''))[0]
                for key, val in filename.items()}

    # Label properties

    def within_context(self, time, direction):
        """Checks if a timestep is within context

        Parameters
        ----------
        time : int
            Timestep to be checked
        direction : str
            Context direction [fwd,bwd]

        Returns
        -------
        bool
            True if timestep is within context, False otherwise

        Raises
        ------
        ValueError
            Invalid context direction
        """
        if len(self.context) == 0:
            return False
        if direction == 'bwd':
            return time > (self.context[0] if self.context[0] < 0 else 0)
        elif direction == 'fwd':
            return time < (self.context[-1] if self.context[-1] > 0 else 0)
        else:
            raise ValueError('Invalid context direction')

    @property
    def with_rgb(self):
        return True

    @property
    def with_rgb_context(self):
        return True

    @property
    def with_intrinsics(self):
        return 'intrinsics' in self.labels

    @property
    def with_dummy_intrinsics(self):
        return 'dummy_intrinsics' in self.labels

    @property
    def with_intrinsics_context(self):
        return 'intrinsics' in self.labels_context

    @property
    def with_dummy_intrinsics_context(self):
        return 'dummy_intrinsics' in self.labels_context

    @property
    def with_depth(self):
        return 'depth' in self.labels

    @property
    def with_dummy_depth(self):
        return 'dummy_depth' in self.labels

    @property
    def with_depth_context(self):
        return 'depth' in self.labels_context

    @property
    def with_dummy_depth_context(self):
        return 'dummy_depth' in self.labels_context

    @property
    def with_pose(self):
        return 'pose' in self.labels

    @property
    def with_dummy_pose(self):
        return 'dummy_pose' in self.labels

    @property
    def with_pose_context(self):
        return 'pose' in self.labels_context

    @property
    def with_dummy_pose_context(self):
        return 'dummy_pose' in self.labels_context

    @property
    def with_extrinsics(self):
        return 'extrinsics' in self.labels

    @property
    def with_extrinsics_context(self):
        return 'extrinsics' in self.labels_context

    @property
    def with_optical_flow(self):
        return 'optical_flow' in self.labels

    @property
    def with_optical_flow_context(self):
        return 'optical_flow' in self.labels_context

    @property
    def with_scene_flow(self):
        return 'scene_flow' in self.labels

    @property
    def with_scene_flow_context(self):
        return 'scene_flow' in self.labels_context

    @property
    def with_motion_mask(self):
        return 'motion_mask' in self.labels

    @property
    def with_valid_optical_flow(self):
        return 'valid_optical_flow' in self.labels

    @property
    def with_valid_optical_flow_context(self):
        return 'valid_optical_flow' in self.labels_context

    @property
    def with_semantic(self):
        return 'semantic' in self.labels

    @property
    def with_semantic_context(self):
        return 'semantic' in self.labels_context
