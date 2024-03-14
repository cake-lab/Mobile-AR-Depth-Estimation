# Copyright 2023 Toyota Research Institute.  All rights reserved.

import glob
import os

import numpy as np

from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.dataloaders.KITTIDataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from efm_datasets.dataloaders.utils.misc import \
    invert_pose, update_dict
from efm_datasets.utils.read import read_image
from PIL import Image

# Cameras from the stereo pair (left is the origin)
IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}


# Name of different calibration files
CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}


PNG_DEPTH_DATASETS = ['groundtruth']
OXTS_POSE_DATA = 'oxts'


def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)


def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(Image.open(file), dtype=int)
    assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)


class KITTIDataset(BaseDataset):
    """KITTI dataset class. 
    
    https://www.cvlibs.net/datasets/kitti/

    Parameters
    ----------
    depth_type : str, optional
        Depth map type, by default None
    single_intrinsics : bool, optional
        True if average dataset intrinsics are to be returned, by default False
    """
    def __init__(self, depth_type=None, single_intrinsics=False, **kwargs):
        super().__init__(**kwargs, base_tag='kitti')

        # Store variables

        self.baseline = + 0.5407
        self.backward_context_paths = []
        self.forward_context_paths = []

        self.single_intrinsics = None if not single_intrinsics else \
            np.array([[0.58, 0.00, 0.5],
                      [0.00, 1.92, 0.5],
                      [0.00, 0.00, 1.0]], dtype=np.float32)

        self.depth_type = depth_type

        self._cache = {}
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

        with open(os.path.join(self.path, self.split), "r") as f:
            data = f.readlines()

        self.paths = []
        # Get file list from data
        for i, fname in enumerate(data):
            path = os.path.join(self.path, fname.split()[0])
            add_flag = True
            if add_flag and self.with_depth:
                # Check if depth file exists
                depth = self._get_depth_file(path, self.depth_type)
                add_flag = depth is not None and os.path.exists(depth)
            if add_flag:
                self.paths.append(path)

        # If using context, filter file list
        if self.with_context:
            paths_with_context = []
            for stride in [1]:
                for idx, file in enumerate(self.paths):
                    backward_context_idxs, forward_context_idxs = \
                        self._get_sample_context(
                            file, self.bwd_context, self.fwd_context, stride)
                    if backward_context_idxs is not None and forward_context_idxs is not None:
                        exists = True
                        if self.with_depth_context:
                            _, depth_context_files = self._get_context_files(
                                self.paths[idx], backward_context_idxs + forward_context_idxs)
                            for depth_file in depth_context_files:
                                exists = os.path.exists(depth_file)
                                if not exists:
                                    break
                        if exists:
                            paths_with_context.append(self.paths[idx])
                            self.forward_context_paths.append(forward_context_idxs)
                            self.backward_context_paths.append(backward_context_idxs[::-1])
            self.paths = paths_with_context

        if len(self.cameras) > 1:
            self.paths = [im.replace('image_03', 'image_02') for im in self.paths]

        if 1 in self.cameras:
            self.paths_stereo = [im.replace('image_02', 'image_03') for im in self.paths]
        else:
            self.paths_stereo = None

    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))
        return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../.."))

    @staticmethod
    def _get_intrinsics(image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return intrinsics
            if IMAGE_FOLDER[cam] in image_file:
                return np.reshape(calib_data[IMAGE_FOLDER[cam].replace('image', 'P_rect')], (3, 4))[:, :3]

    @staticmethod
    def _read_raw_calib_file(folder):
        """Read raw calibration files from folder."""
        return read_calib_file(os.path.join(folder, CALIB_FILE['cam2cam']))

    @staticmethod
    def _get_keypoints(filename, size):
        """Get keypoints from image file."""
        filename = filename. \
            replace('KITTI_tiny', 'KITTI_tiny_keypoints'). \
            replace('.png', '.txt.npz')
        keypoints = np.load(filename)['data']
        keypoints_coord, keypoints_desc = keypoints[:, :2], keypoints[:, 2:]
        keypoints_coord[:, 0] *= size[0] / 320
        keypoints_coord[:, 1] *= size[1] / 240
        return keypoints_coord, keypoints_desc

    def get_filename(self, sample_idx):
        """Returns the filename for an index."""
        filename = os.path.splitext(self.paths[sample_idx].replace(self.path + '/', ''))[0]
        for cam in ['left', 'right']:
            filename = filename.replace('{}/data'.format(IMAGE_FOLDER[cam]),
                                        'proj_depth/{}/%s' % IMAGE_FOLDER[cam])
        return filename

    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if depth_file.endswith('.npz'):
            return read_npz_depth(depth_file, 'velodyne')
        elif depth_file.endswith('.png'):
            return read_png_depth(depth_file)
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    @staticmethod
    def _get_depth_file(image_file, depth_type):
        """Get the corresponding depth file from an image file."""
        for cam in ['left', 'right']:
            if IMAGE_FOLDER[cam] in image_file:
                depth_file = image_file.replace(
                    IMAGE_FOLDER[cam] + '/data', 'proj_depth/{}/{}'.format(depth_type, IMAGE_FOLDER[cam]))
                if depth_type not in PNG_DEPTH_DATASETS:
                    depth_file = depth_file.replace('png', 'npz')
                return depth_file

    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """Get sample context"""
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = int(base)

        # Check number of files in folder
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
        else:
            max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
            self._cache[parent_folder] = max_num_files

        # Check bounds
        if (f_idx - backward_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            return None, None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > 0:
            c_idx -= stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None, None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx >= max_num_files:
            return None, None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """Returns image and depth context files"""
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        if self.with_depth:
            depth_context_paths = [self._get_depth_file(f, self.depth_type) for f in image_context_paths]
            return image_context_paths, depth_context_paths
        else:
            return image_context_paths, None

    def _get_imu2cam_transform(self, image_file):
        """Gets the transformation between IMU an camera from an image file"""
        parent_folder = self._get_parent_folder(image_file)
        if image_file in self.imu2velo_calib_cache:
            return self.imu2velo_calib_cache[image_file]

        cam2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['cam2cam']))
        imu2velo = read_calib_file(os.path.join(parent_folder, CALIB_FILE['imu2velo']))
        velo2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['velo2cam']))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
        self.imu2velo_calib_cache[image_file] = imu2cam
        return imu2cam

    @staticmethod
    def _get_oxts_file(image_file):
        """Gets the oxts file from an image file."""
        # find oxts pose file
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return file name
            if IMAGE_FOLDER[cam] in image_file:
                return image_file.replace(IMAGE_FOLDER[cam], OXTS_POSE_DATA).replace('.png', '.txt')
        # Something went wrong (invalid image file)
        raise ValueError('Invalid KITTI path for pose supervision.')

    def _get_oxts_data(self, image_file):
        """Gets the oxts data from an image file."""
        oxts_file = self._get_oxts_file(image_file)
        if oxts_file in self.oxts_cache:
            oxts_data = self.oxts_cache[oxts_file]
        else:
            oxts_data = np.loadtxt(oxts_file, delimiter=' ', skiprows=0)
            self.oxts_cache[oxts_file] = oxts_data
        return oxts_data

    def _get_pose(self, image_file, camera):
        """Gets the pose information from an image file."""
        if image_file in self.pose_cache:
            return self.pose_cache[image_file]
        # Find origin frame in this sequence to determine scale & origin translation
        base, ext = os.path.splitext(os.path.basename(image_file))
        origin_frame = os.path.join(os.path.dirname(image_file), str(0).zfill(len(base)) + ext)
        # Get origin data
        origin_oxts_data = self._get_oxts_data(origin_frame)
        lat = origin_oxts_data[0]
        scale = np.cos(lat * np.pi / 180.)
        # Get origin pose
        origin_R, origin_t = pose_from_oxts_packet(origin_oxts_data, scale)
        origin_pose = transform_from_rot_trans(origin_R, origin_t)
        # Compute current pose
        oxts_data = self._get_oxts_data(image_file)
        R, t = pose_from_oxts_packet(oxts_data, scale)
        pose = transform_from_rot_trans(R, t)
        # Compute odometry pose
        imu2cam = self._get_imu2cam_transform(image_file)
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose @ np.linalg.inv(imu2cam)).astype(np.float32)
        odo_pose = invert_pose(odo_pose)
        # Cache and return pose
        self.pose_cache[image_file] = odo_pose
        if camera == 1:
            odo_pose[0, -1] -= self.baseline
        return odo_pose

    def __len__(self):
        """Dataset length."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""

        # Initialize sample
        sample, idx = self.initialize_sample(idx)

        # Loop over all requested cameras
        for cam_idx, cam in enumerate(self.cameras):

            # Get target filename and prepare key tuple
            filename = self.paths[idx] if cam == 0 else self.paths_stereo[idx]
            time_cam = (0, cam_idx)
            update_dict(sample, 'filename', time_cam, filename)

            # Add information to the sample dictionary based on label request
            update_dict(sample, 'rgb', (0, cam_idx), read_image(filename))
            if self.with_intrinsics:
                if self.single_intrinsics is not None:
                    intrinsics = self.single_intrinsics.copy()
                    intrinsics[0, :] *= sample['rgb'][time_cam].size[0]
                    intrinsics[1, :] *= sample['rgb'][time_cam].size[1]
                    update_dict(sample, 'intrinsics', time_cam, intrinsics)
                else:
                    parent_folder = self._get_parent_folder(filename)
                    if parent_folder in self.calibration_cache:
                        c_data = self.calibration_cache[parent_folder]
                    else:
                        c_data = self._read_raw_calib_file(parent_folder)
                        self.calibration_cache[parent_folder] = c_data
                    update_dict(sample, 'intrinsics', time_cam,
                                self._get_intrinsics(filename, c_data))
            if self.with_pose:
                update_dict(sample, 'pose', time_cam,
                            self._get_pose(filename, cam))
            if self.with_depth:
                update_dict(sample, 'depth', time_cam,
                            self._read_depth(self._get_depth_file(filename, self.depth_type)))

            # If includes context
            if self.with_context:
                # Get context filenames and loop over them
                all_context_idxs = self.backward_context_paths[idx] + self.forward_context_paths[idx]
                image_context_paths, depth_context_paths = self._get_context_files(filename, all_context_idxs)
                for time, filename in zip(self.context, image_context_paths):
                    time_cam = (time, cam_idx)
                    update_dict(sample, 'filename', time_cam, filename)
                    update_dict(sample, 'rgb', time_cam,
                                read_image(filename))
                if self.with_intrinsics_context:
                    for time, filename in zip(self.context, image_context_paths):
                        time_cam = (time, cam_idx)
                        update_dict(sample, 'intrinsics', time_cam,
                                    sample['intrinsics'][(0, cam_idx)])
                if self.with_pose_context:
                    for time, filename in zip(self.context, image_context_paths):
                        time_cam = (time, cam_idx)
                        update_dict(sample, 'pose', time_cam,
                                    self._get_pose(filename, cam))
                if self.with_depth_context:
                    for time, filename in zip(self.context, depth_context_paths):
                        time_cam = (time, cam_idx)
                        update_dict(sample, 'depth', time_cam,
                                    self._read_depth(self._get_depth_file(filename, self.depth_type)))

        # Return post-processed sample
        return self.post_process_sample(sample)
