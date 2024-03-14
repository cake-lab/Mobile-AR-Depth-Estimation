
import csv
import os

import cv2
import numpy as np

from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.dataloaders.utils.FolderTree import FolderTree
from efm_datasets.dataloaders.utils.misc import make_relative_pose, update_dict, update_dict_nested, invert_pose
from efm_datasets.utils.read import read_image


def make_tree(path, sub_folder, camera, mode, context):
    """Create a FolderTree with proper folder structure"""
    path = os.path.join(path, sub_folder)
    sub_folders = '{}/frames/{}/Camera_{}'.format(mode, sub_folder, camera)
    return FolderTree(path, sub_folders=[sub_folders], context=context)


class VKITTI2Dataset(BaseDataset):
    """VKITTI2 dataset. 
    
    https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

    Parameters
    ----------
    zero_origin : bool, optional
        True if poses are relative to the beginning of the sequence, by default False
    num_images : int, optional
        How many images should be considered (useful for debugging and overfitting), by default None
    """    
    def __init__(self, zero_origin=False, num_images=None, **kwargs):
        super().__init__(**kwargs, base_tag='vkitti2')

        # Store variables
        self.mode = 'clone'
        self.num_images = num_images

        # Create data tree
        self.rgb_tree = make_tree(
            self.path, 'rgb', 0, self.mode, self.context)

        # If zero_origin is requested, the first sample of each scene is set to identity
        if zero_origin:
            sample = self.__getitem__(0, force_camera=0)
            self.base_pose = invert_pose(sample['pose'][(0,0)].cpu().numpy())

    def __len__(self):
        """Dataset length"""
        return len(self.rgb_tree) if self.num_images is None else \
            min(len(self.rgb_tree), self.num_images)

    @staticmethod
    def get_rgb(filename):
        """Get image from filename."""
        return read_image(filename)

    @staticmethod
    def get_depth(filename):
        """Get depth from filename."""
        filename = filename.replace('rgb', 'depth').replace('jpg', 'png')
        return cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.

    @staticmethod
    def get_intrinsics(filename, camera, mode):
        """Get intrinsics from filename."""
        # Get sample number in the scene
        number = int(filename.split('/')[-1].replace('rgb_', '').replace('.jpg', ''))
        # Get intrinsic filename
        filename_idx = filename.rfind(mode) + len(mode)
        filename_intrinsics = os.path.join(filename[:filename_idx].replace(
            '/rgb/', '/textgt/'), 'intrinsic.txt')
        # Open intrinsic file
        with open(filename_intrinsics, 'r') as f:
            # Get intrinsic parameters
            lines = list(csv.reader(f, delimiter=' '))[1:]
            params = [float(p) for p in lines[number * 2 + camera][2:]]
            # Build intrinsics matrix
            intrinsics = np.array([[params[0], 0.0, params[2]],
                                   [0.0, params[1], params[3]],
                                   [0.0, 0.0, 1.0]]).astype(np.float32)
        # Return intrinsics
        return intrinsics

    @staticmethod
    def get_pose(filename, camera, mode):
        """Get pose from filename."""
        # Get sample number in the scene
        number = int(filename.split('/')[-1].replace('rgb_', '').replace('.jpg', ''))
        # Get intrinsic filename
        filename_idx = filename.rfind(mode) + len(mode)
        filename_pose = os.path.join(filename[:filename_idx].replace(
            '/rgb/', '/textgt/'), 'extrinsic.txt')
        # Open intrinsics file
        with open(filename_pose, 'r') as f:
            # Get pose parameters
            lines = list(csv.reader(f, delimiter=' '))[1:]
            pose = np.array([float(p) for p in lines[number * 2 + camera][2:]]).reshape(4, 4)
        # Return pose
        return pose

    @staticmethod
    def get_optical_flow(filename, direction):
        """Get optical flow from filename."""
        # Get filename
        if direction == 'bwd':
            filename = filename.replace('rgb', 'backwardFlow')
        elif direction == 'fwd':
            filename = filename.replace('/rgb/', '/forwardFlow/').replace('rgb_', 'flow_')
        else:
            raise ValueError('Invalid optical flow mode')
        filename = filename.replace('jpg', 'png')
        # Return None if file does not exist
        if not os.path.exists(filename):
            return None
        else:
            # Get optical flow
            optical_flow = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            h, w = optical_flow.shape[:2]
            # Get invalid optical flow pixels
            invalid = optical_flow[..., 0] == 0
            # Normalize and scale optical flow values
            optical_flow = 2.0 / (2 ** 16 - 1.0) * optical_flow[..., 2:0:-1].astype('f4') - 1.
            optical_flow[..., 0] *= w - 1
            optical_flow[..., 1] *= h - 1
            # Remove invalid pixels
            optical_flow[invalid] = 0
            return optical_flow

    @staticmethod
    def get_scene_flow(filename, mode):
        """Get scene flow from filename."""
        # Get filename
        if mode == 'bwd':
            filename = filename.replace('rgb', 'backwardSceneFlow')
        elif mode == 'fwd':
            filename = filename.replace('/rgb/', '/forwardSceneFlow/').replace('rgb_', 'sceneFlow_')
        else:
            raise ValueError('Invalid scene flow mode')
        filename = filename.replace('jpg', 'png')
        # Return None if file does not exist
        if not os.path.exists(filename):
            return None
        else:
            # Get scene flow
            scene_flow = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # Return normalized and scaled optical flow (-10m to 10m)
            return (scene_flow[:, :, ::-1] * 2. / 65535. - 1.) * 10.

    def __getitem__(self, idx, force_camera=None):
        """Get dataset sample given an index."""

        # Initialize sample
        sample, idx = self.initialize_sample(idx)

        # Loop over all requested cameras
        for cam_idx, cam in enumerate(self.cameras if force_camera is None else [force_camera]):

            # Get target filename and prepare key tuple
            filename = self.rgb_tree.get_item(idx)[0]
            filename = filename.replace('Camera_0', 'Camera_{}'.format(cam))
            time_cam = (0, cam_idx)
            update_dict(sample, 'filename', time_cam, filename)

            # Add information to the sample dictionary based on label request
            update_dict(sample, 'rgb', time_cam,
                        self.get_rgb(filename))
            if self.with_intrinsics:
                update_dict(sample, 'intrinsics', time_cam,
                            self.get_intrinsics(filename, cam, self.mode))
            if self.with_pose:
                update_dict(sample, 'pose', time_cam,
                            self.get_pose(filename, cam, self.mode))
            if self.with_depth:
                update_dict(sample, 'depth', time_cam,
                            self.get_depth(filename))
            if self.with_optical_flow:
                update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] - 1, cam_idx),
                                   self.get_optical_flow(filename, 'bwd'))
                update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] + 1, cam_idx),
                                   self.get_optical_flow(filename, 'fwd'))
            if self.with_scene_flow:
                update_dict_nested(sample, 'scene_flow', time_cam, (time_cam[0] - 1, cam_idx),
                                   self.get_scene_flow(filename, 'bwd'))
                update_dict_nested(sample, 'scene_flow', time_cam, (time_cam[0] + 1, cam_idx),
                                   self.get_scene_flow(filename, 'fwd'))

            if self.with_context:
                # Get context filenames and loop over them
                filename_context = self.rgb_tree.get_context(idx)
                for time, filename in filename_context.items():
                    filename = filename.replace('Camera_0', 'Camera_{}'.format(cam))
                    time_cam = (time, cam_idx)
                    update_dict(sample, 'filename', time_cam, filename)
                    # Add information to the sample dictionary based on label request
                    update_dict(sample, 'rgb', time_cam,
                                self.get_rgb(filename))
                    if self.with_intrinsics_context:
                        update_dict(sample, 'intrinsics', time_cam,
                                    self.get_intrinsics(filename, cam, self.mode))
                    if self.with_pose_context:
                        update_dict(sample, 'pose', time_cam,
                                    self.get_pose(filename, cam, self.mode))
                    if self.with_depth_context:
                        update_dict(sample, 'depth', time_cam,
                                    self.get_depth(filename))
                    if self.with_optical_flow_context:
                        if self.within_context(time, 'bwd'):
                            update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] - 1, cam_idx),
                                               self.get_optical_flow(filename, 'bwd'))
                        if self.within_context(time, 'fwd'):
                            update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] + 1, cam_idx),
                                               self.get_optical_flow(filename, 'fwd'))
                    if self.with_scene_flow_context:
                        if self.within_context(time, 'bwd'):
                            update_dict_nested(sample, 'scene_flow', time_cam, (time_cam[0] - 1, cam_idx),
                                               self.get_scene_flow(filename, 'bwd'))
                        if self.within_context(time, 'fwd'):
                            update_dict_nested(sample, 'scene_flow', time_cam, (time_cam[0] + 1, cam_idx),
                                               self.get_scene_flow(filename, 'fwd'))

        # Return post-processed sample
        return self.post_process_sample(sample)

