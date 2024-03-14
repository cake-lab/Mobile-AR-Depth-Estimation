from collections import defaultdict
import os
from PIL import Image

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion

from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.dataloaders.utils.misc import make_relative_pose, invert_pose


SCENES = {
    "train": splits.train,
    "val": splits.val,
    "trainval": splits.train + splits.val,
    "test": splits.test,
    "mini-train": splits.mini_train,
    "mini-val": splits.mini_val
}

CAMERAS = {
    0: "CAM_FRONT",
    1: "CAM_FRONT_RIGHT",
    2: "CAM_BACK_RIGHT",
    3: "CAM_BACK",
    4: "CAM_BACK_LEFT",
    5: "CAM_FRONT_LEFT"
}


class NuScenesDataset(BaseDataset):
    """NuScenes dataset class.

    https://www.nuscenes.org/

    Parameters
    ----------
    keyframe_only : bool, optional
        True if only labeled frames are returned, by default False
    """
    def __init__(self, keyframe_only=False, **kwargs):
        super().__init__(**kwargs, base_tag='nuscenes')

        assert self.split in {
            "train", "val", "trainval", "test", "mini-train", "mini-val"
        }
        self.scenes = SCENES[self.split]
        self.keyframe_only = keyframe_only
        self.camera_dicts = self.build_index()

    def build_index(self):
        """Builds an index of all the camera frames in the dataset."""
        # create nuScenes official devkit handle
        if 'mini' in self.split:
            version = 'v1.0-mini'
        elif 'test' in self.split:
            version = 'v1.0-test'
        else:  # "train", "val" or "trainval" split
            version = 'v1.0-trainval'
        nusc = NuScenes(version, self.path, verbose=False)

        camera_dicts_all = defaultdict(list)
        scene_tokens = {scene['name']: scene['token'] for scene in nusc.scene}
        for scene_name in self.scenes:
            scene = nusc.get('scene', scene_tokens[scene_name])
            first_sample = nusc.get('sample', scene['first_sample_token'])

            for cam in self.cameras:
                camera_dicts = []
                cam_name = CAMERAS[cam]
                cam_meta = nusc.get('sample_data',
                                    first_sample['data'][cam_name])
                cam_dict = self.read_cam_meta(cam_meta, nusc)
                camera_dicts.append(cam_dict)
                while cam_meta['next']:
                    cam_meta = nusc.get('sample_data', cam_meta['next'])
                    cam_dict = self.read_cam_meta(cam_meta, nusc)
                    if (self.keyframe_only and cam_dict['is_key_frame']) or (
                            not self.keyframe_only):
                        camera_dicts.append(cam_dict)

                # remove first/last few frames that don't have complete context
                camera_dicts = camera_dicts[self.bwd_context:len(camera_dicts) - self.fwd_context]

                camera_dicts_all[cam].extend(camera_dicts)

        return camera_dicts_all

    def __len__(self):
        """Dataset length."""
        return len(list(self.camera_dicts.values())[0]) - (self.bwd_context + self.fwd_context)

    @staticmethod
    def compose_transformation(rot, tvec):
        """Reorientate and translate pose."""
        pose = np.concatenate([rot, tvec[:, None]], axis=1)
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
        return pose

    def read_cam_meta(self, cam_meta, nusc):
        """Parse camera metadata."""
        cam_dict = dict()
        cam_dict['filename'] = cam_meta.get('filename', None)
        cam_dict['sensor'] = nusc.get('calibrated_sensor',
                                      cam_meta['calibrated_sensor_token'])
        cam_dict['ego_pose'] = nusc.get('ego_pose', cam_meta['ego_pose_token'])
        cam_dict['is_key_frame'] = cam_meta.get('is_key_frame', False)

        if cam_dict['is_key_frame']:
            sample = nusc.get('sample', cam_meta['sample_token'])
            lidar_meta = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_dict = dict()
            lidar_dict['filename'] = lidar_meta.get('filename', None)
            lidar_dict['sensor'] = nusc.get(
                'calibrated_sensor', lidar_meta['calibrated_sensor_token'])
            lidar_dict['ego_pose'] = nusc.get('ego_pose',
                                              lidar_meta['ego_pose_token'])
            cam_dict['lidar_dict'] = lidar_dict

        return cam_dict

    def get_point_cloud(self, cam_dict):
        """Parse and return pointcloud"""
        lidar_dict = cam_dict['lidar_dict']

        # read point cloud
        pc_file = os.path.join(self.path, lidar_dict['filename'])
        pc = np.fromfile(pc_file, np.float32).reshape(-1, 5)
        pc = pc[..., :3]

        pose = self.get_sensor_pose(lidar_dict)

        return pc, pose

    @staticmethod
    def project_depth(cam_pose, lidar_pose, intrinsics, points, H, W):
        """Project depth to image plane."""
        if points.shape[-1] == 4:
            points[..., -1] = 1.0
        if points.shape[-1] == 3:
            points = np.concatenate(
                [points, np.ones_like(points[..., :1])], axis=-1)

        lidar2cam = np.matmul(np.linalg.inv(cam_pose), lidar_pose)
        points = np.matmul(lidar2cam, points.transpose())  # (4, N)
        uvs = np.matmul(intrinsics, points[:3, :])
        uvs = uvs / (uvs[2:3, :] + 1e-6)
        uvs = uvs[:2, :]

        depth_map = np.zeros((H, W), dtype=points.dtype)
        valid_mask = points[2, :] > 0
        valid_pts = points[:, valid_mask]
        uvs = uvs[:, valid_mask]

        valid_mask = (uvs[0] > 0) & (uvs[0] < W) & (uvs[1] > 0) & (uvs[1] < H)
        valid_pts = valid_pts[:, valid_mask]
        uvs = uvs[:, valid_mask].astype(int)

        depth_map[uvs[1], uvs[0]] = valid_pts[2]

        return depth_map

    def get_sensor_pose(self, sensor_meta):
        """Parse and return sensor pose."""
        sensor = sensor_meta['sensor']
        s2e_tvec = np.array(sensor['translation'])
        s2e_rot = Quaternion(sensor['rotation']).rotation_matrix
        s2e_pose = self.compose_transformation(s2e_rot, s2e_tvec)

        # global pose
        ego_pose = sensor_meta['ego_pose']
        e2g_tvec = np.array(ego_pose['translation'])
        e2g_rot = Quaternion(ego_pose['rotation']).rotation_matrix
        e2g_pose = self.compose_transformation(e2g_rot, e2g_tvec)
        s2g_pose = np.matmul(e2g_pose, s2e_pose)

        return s2g_pose

    def get_camera_data(self, cam_dict, is_context=False):
        """Parse and return camera data."""
        sample = dict()

        # rgb
        filename = os.path.join(self.path, cam_dict['filename'])
        sample['rgb'] = Image.open(filename)
        W, H = sample['rgb'].size
        sample['filename'] = filename[:-4].replace(self.path + '/', '')

        # intrinsics & pose in ego body frame
        sensor = cam_dict['sensor']
        intrinsics = np.array(sensor['camera_intrinsic'])
        sample['intrinsics'] = intrinsics
        sample['pose'] = self.get_sensor_pose(cam_dict)

        if (not is_context) and self.with_depth:
            if cam_dict['is_key_frame']:
                points, lidar_pose = self.get_point_cloud(cam_dict)
                sample['depth'] = self.project_depth(sample['pose'],
                                                     lidar_pose, intrinsics,
                                                     points, H, W)

        if is_context and self.with_depth_context:
            if cam_dict['is_key_frame']:
                points, lidar_pose = self.get_point_cloud(cam_dict)
                sample['depth'] = self.project_depth(sample['pose'],
                                                     lidar_pose, intrinsics,
                                                     points, H, W)

        return sample

    def __getitem__(self, idx):
        """Get dataset sample given an index."""

        # Initialize sample
        sample, idx = self.initialize_sample(idx)
        idx = idx + self.bwd_context

        for cam_idx, cam in enumerate(self.cameras):

            # get target image
            cam_dict = self.camera_dicts[cam][idx]

            sample_target = self.get_camera_data(cam_dict)
            for key, val in sample_target.items():
                if key not in sample:
                    sample[key] = dict()
                sample[key][(0, cam_idx)] = val

            # get context images
            for ctx in self.context:
                cam_dict = self.camera_dicts[cam][idx + ctx]
                sample_context = self.get_camera_data(cam_dict, is_context=True)
                for key, val in sample_context.items():
                    if key not in sample:
                        sample[key] = dict()
                    sample[key][(ctx, cam_idx)] = val

        for key in sample['pose']:
            sample['pose'][key] = invert_pose(sample['pose'][key])

        if not self.with_pose:
            del sample['pose']

        # Return post-processed sample
        return self.post_process_sample(sample)
