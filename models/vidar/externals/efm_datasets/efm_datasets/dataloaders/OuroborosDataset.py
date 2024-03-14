# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os
import pickle
from collections import OrderedDict

import numpy as np
from ouroboros.dgp.utils.camera import Camera
from ouroboros.dgp.utils.pose import Pose
from efm_datasets.dataloaders.utils.misc import update_dict, update_dict_nested

from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.utils.data import make_list
from efm_datasets.utils.read import read_image, read_pickle, read_numpy
from efm_datasets.utils.types import is_str
from efm_datasets.utils.write import write_pickle
from efm_datasets.utils.types import is_dict


def merge_samples(samples):
    merged_sample = {}
    for sample in samples:
        for key1 in sample.keys():
            if key1 not in merged_sample.keys():
                merged_sample[key1] = {}
            if is_dict(sample[key1]):
                for key2 in sample[key1].keys():
                    merged_sample[key1][key2] = sample[key1][key2]
            else:
                merged_sample[key1] = sample[key1]
    return merged_sample


def load_from_file(filename, key):
    """Load data cache from a file"""
    data = read_numpy(filename, allow_pickle=True)[key]
    if len(data.shape) == 0:
        data = None
    return data


def save_to_file(filename, key, value):
    """Save data to a cache file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, **{key: value})


def get_points_inside_bbox3d(points, bboxes3d):
    """
    Returns point indices inside bounding boxes

    Parameters
    ----------
    points : np.array (Nx3)
        Points to be checked (unordered, usually from LiDAR or lifted monocular)
    bboxes3d : list of np.array
        Corners of bounding boxes to be checked (as given by DGP's .corner 3d bounding box property)

    Returns
    -------
    points : np.array (Nx3)
        Points inside bounding boxes
    """
    indices = []
    for bbox in bboxes3d:
        p1, p2, p3, p4 = np.array(bbox)[[0, 1, 3, 4]]
        d12, d14, d15 = p1 - p2, p1 - p3, p1 - p4

        u = np.expand_dims(np.cross(d14, d15), 1)
        v = np.expand_dims(np.cross(d12, d14), 1)
        w = np.expand_dims(np.cross(d12, d15), 1)

        p1 = np.expand_dims(p1, 0)
        p2 = np.expand_dims(p2, 0)
        p3 = np.expand_dims(p3, 0)
        p4 = np.expand_dims(p4, 0)

        pdotu = np.dot(points, u)
        pdotv = np.dot(points, v)
        pdotw = np.dot(points, w)

        idx = (pdotu < np.dot(p1, u)) & (pdotu > np.dot(p2, u)) & \
              (pdotv < np.dot(p1, v)) & (pdotv > np.dot(p4, v)) & \
              (pdotw > np.dot(p1, w)) & (pdotw < np.dot(p3, w))
        indices.append(idx.nonzero()[0])
    return np.concatenate([points[idx] for idx in indices], 0)


def _prepare_ouroboros_ontology(ontology):
    """Read and prepare ontology to return as a data field"""
    name = ontology.contiguous_id_to_name
    colormap = ontology.contiguous_id_colormap
    output = OrderedDict()
    for key in name.keys():
        output[key] = {'name': name[key], 'color': np.array(colormap[key])}
    return output


def generate_proj_maps(camera, Xw, shape, bwd_scene_flow=None, fwd_scene_flow=None):
    """Render pointcloud on image.

    Parameters
    ----------
    camera: Camera
        Camera object with appropriately set extrinsics wrt world.

    Xw: np.ndarray (N x 3)
        3D point cloud (x, y, z) in the world coordinate.

    shape: np.ndarray (H, W)
        Output depth image shape.

    bwd_scene_flow: np.array
        Backward scene flow for projection (H, W, 3)

    fwd_scene_flow: np.array
        Forward scene flow for projection (H, W, 3)

    Returns
    -------
    depth: np.array
        Rendered depth image.
    """
    assert len(shape) == 2, 'Shape needs to be 2-tuple.'
    # Move point cloud to the camera's (C) reference frame from the world (W)
    Xc = camera.p_cw * Xw
    if fwd_scene_flow is not None:
        fwd_scene_flow = camera.p_cw * (Xw + fwd_scene_flow) - Xc
    if bwd_scene_flow is not None:
        bwd_scene_flow = camera.p_cw * (Xw + bwd_scene_flow) - Xc
    # Project the points as if they were in the camera's frame of reference
    uv = Camera(K=camera.K).project(Xc).astype(int)
    # Colorize the point cloud based on depth
    z_c = Xc[:, 2]

    # Create an empty image to overlay
    H, W = shape
    proj_depth = np.zeros((H, W), dtype=np.float32)
    in_view = np.logical_and.reduce([(uv >= 0).all(axis=1), uv[:, 0] < W, uv[:, 1] < H, z_c > 0])
    uv, z_c = uv[in_view], z_c[in_view]
    proj_depth[uv[:, 1], uv[:, 0]] = z_c

    # Project scene flow into image plane
    proj_bwd_scene_flow = proj_fwd_scene_flow = None
    if bwd_scene_flow is not None:
        proj_bwd_scene_flow = np.zeros((H, W, 3), dtype=np.float32)
        proj_bwd_scene_flow[uv[:, 1], uv[:, 0]] = bwd_scene_flow[in_view]
    if fwd_scene_flow is not None:
        proj_fwd_scene_flow = np.zeros((H, W, 3), dtype=np.float32)
        proj_fwd_scene_flow[uv[:, 1], uv[:, 0]] = fwd_scene_flow[in_view]

    # Return projected maps
    return proj_depth, proj_bwd_scene_flow, proj_fwd_scene_flow


class OuroborosDataset(BaseDataset):
    """Ouroboros dataset class. 

    https://github.com/TRI-ML/dgp

    Parameters
    ----------
    split : str
        Which split to use [train,val,test]
    tag : str, optional
        Dataset tag, to identify its samples in a batch, by default None
    depth_type : str, optional
        Depth map type, by default None
    input_depth_type : str, optional
        Input depth map type (used for depth completion), by default None
    mask_rgb : str, optional
        Path to image mask, by default None
    virtual : bool, optional
        True if the dataset is virtual (e.g., Parallel Domain), by default False
    save_data : bool, optional
        True if sample data is stored for later use, by default False
    """
    def __init__(self, split, tag=None,
                 depth_type=None, input_depth_type=None,
                 mask_rgb=None, virtual=False, save_data=False, **kwargs):
        super().__init__(**kwargs)
        self.tag = 'ouroboros' if tag is None else tag

        cameras = [c if is_str(c) else 'camera_%02d' % c for c in self.cameras]

        # Store variables
        self.split = split
        self.dataset_idx = 0
        self.sensors = list(cameras)
        self.virtual = virtual

        # Store task information
        self.depth_type = depth_type
        self.input_depth_type = input_depth_type
        self.only_cache = False
        self.save_data = save_data

        self.mask_rgb = mask_rgb

        # Add requested annotations
        requested_annotations = []

        # Add depth sensor
        if self.with_depth and not self.only_cache and \
                self.depth_type != 'zbuffer':
            self.sensors.append(depth_type)
        self.depth_idx = len(self.sensors) - 1

        # Choose which dataset to use
        if not self.virtual:
            from ouroboros.dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
            dataset = SynchronizedSceneDataset
            extra_args = {}
        else:
            from ouroboros.dgp.datasets.pd_dataset import ParallelDomainSceneDataset
            dataset = ParallelDomainSceneDataset
            extra_args = {
                'use_virtual_camera_datums': False,
                # 'return_scene_flow': self.with_scene_flow,
            }

        # Initialize chosen dataset
        self.dataset = dataset(
            scene_dataset_json=self.path,
            split=split,
            datum_names=self.sensors,
            backward_context=self.bwd_context,
            forward_context=self.fwd_context,
            requested_annotations=requested_annotations,
            only_annotated_datums=False,
            **extra_args,
        )

    def save_data_fn(self, filename, sample, tgt=(0, 0)):
        """Save data for later use

        Parameters
        ----------
        filename : str
            Fileanme to save data
        sample : dict
            Sample data
        tgt : tuple, optional
            Timestep and camera indexes, by default (0, 0)
        """
        path = '/'.join(self.path.split('/')[:-1])

        if 'intrinsics' in sample:
            filename_intrinsics = filename.format('intrinsics')
            path_intrinsics = os.path.join(path, filename_intrinsics + '.npy')
            os.makedirs(os.path.dirname(path_intrinsics), exist_ok=True)
            np.save(path_intrinsics, sample['intrinsics'][tgt])

        if 'pose' in sample:
            filename_pose = filename.format('pose')
            path_pose = os.path.join(path, filename_pose + '.npy')
            os.makedirs(os.path.dirname(path_pose), exist_ok=True)
            np.save(path_pose, sample['pose'][tgt])

        if 'extrinsics' in sample:
            filename_extrinsics = filename.format('extrinsics')
            path_extrinsics = os.path.join(path, filename_extrinsics + '.npy')
            os.makedirs(os.path.dirname(path_extrinsics), exist_ok=True)
            np.save(path_extrinsics, sample['extrinsics'][tgt])

    def depth_to_world_points(self, depth, datum_idx):
        """
        Unproject depth from a camera's perspective into a world-frame pointcloud

        Parameters
        ----------
        depth : np.array (HxW)
            Depth map to be lifted
        datum_idx : int
            Index of the camera

        Returns
        -------
        pointcloud : np.array (Nx3)
            Lifted 3D pointcloud
        """
        # Access data
        intrinsics = self.get_current('intrinsics', datum_idx)
        pose = self.get_current('pose', datum_idx)
        # Create pixel grid for 3D unprojection
        h, w = depth.shape[:2]
        uv = np.mgrid[:w, :h].transpose(2, 1, 0).reshape(-1, 2).astype(np.float32)
        # Unproject grid to 3D in the camera frame of reference
        pcl = Camera(K=intrinsics).unproject(uv) * depth.reshape(-1, 1)
        # Return pointcloud in world frame of reference
        return pose * pcl

    def create_camera(self, datum_idx, context=None):
        """
        Create current camera.

        Parameters
        ----------
        datum_idx : int
            Index of the camera
        context : int
            Context value for choosing current of reference information

        Returns
        -------
        camera : Camera
            DGP camera
        """
        camera_pose = self.get_current_or_context('pose', datum_idx, context)
        camera_intrinsics = self.get_current_or_context('intrinsics', datum_idx, context)
        return Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())

    def get_optical_flow(self, filename, direction):
        """
        Get optical flow from a filename (only PD)

        Parameters
        ----------
        filename : str
            Optical flow filename
        direction : str
            Direction ['bwd', 'fwd']

        Returns
        -------
        optical_flow : np.array
            Optical flow [H,W,2]
        """
        # Check if direction is valid
        assert direction in ['bwd', 'fwd']
        direction = 'back_motion_vectors_2d' if direction == 'bwd' else 'motion_vectors_2d'
        # Get filename path and load optical flow
        path = os.path.join(os.path.dirname(self.path),
                            filename.format(direction) + '.png')
        if not os.path.exists(path):
            return None
        else:
            optflow = np.array(read_image(path, mode='RGBA'))
            # Convert to uv motion
            dx_i = optflow[..., 0] + optflow[..., 1] * 256
            dy_i = optflow[..., 2] + optflow[..., 3] * 256
            dx = ((dx_i / 65535.0) * 2.0 - 1.0) * optflow.shape[1]
            dy = ((dy_i / 65535.0) * 2.0 - 1.0) * optflow.shape[0]
            # Return stacked array
            return np.stack((dx, dy), 2)

    def get_fwd_optical_flow(self, filename):
        """Get forward optical flow"""
        return self.get_optical_flow(filename, 'fwd')

    def get_bwd_optical_flow(self, filename):
        """Get backwards optical flow"""
        return self.get_optical_flow(filename, 'bwd')

    def create_proj_maps(self, filename, camera_idx, depth_idx, depth_type,
                         world_points=None, context=None, with_scene_flow=False):
        """
        Creates the depth map for a camera by projecting LiDAR information.
        It also caches the depth map following DGP folder structure, so it's not recalculated

        Parameters
        ----------
        filename : str
            Filename used for loading / saving
        camera_idx : int
            Camera sensor index
        depth_idx : int
            Depth sensor index
        depth_type : str
            Which depth type will be loaded
        world_points : np.array (Nx3)
            Points that will be projected (optional)
        context : int
            Context value for choosing current of reference information
        with_scene_flow : bool
            Return scene flow information as well or not

        Returns
        -------
        depth : np.array [H, W]
            Depth map for that datum in that sample
        """
        # If we want the z-buffer (simulation)
        if depth_type == 'zbuffer':
            sensor_name = self.get_current('datum_name', camera_idx)
            filename = filename.replace(self.sensors[camera_idx], sensor_name)
            filename = '{}/{}.npz'.format(
                os.path.dirname(self.path), filename.format('depth'))
            return read_numpy(filename, key='data'), None, None
        # Otherwise, we want projected information
        filename_depth = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('projected/depth/{}'.format(depth_type)))
        filename_bwd_scene_flow = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('projected/bwd_scene_flow/{}'.format(depth_type)))
        filename_fwd_scene_flow = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('projected/fwd_scene_flow/{}'.format(depth_type)))
        # Load and return if exists
        try:
            # Get cached depth map
            depth = load_from_file(filename_depth, 'depth')
            if not with_scene_flow:
                return depth, None, None
            else:
                # Get cached scene flow maps
                bwd_scene_flow = load_from_file(filename_bwd_scene_flow, 'scene_flow')
                fwd_scene_flow = load_from_file(filename_fwd_scene_flow, 'scene_flow')
                return depth, bwd_scene_flow, fwd_scene_flow
        except:
            pass
        # Initialize scene flow maps
        bwd_scene_flow = fwd_scene_flow = None
        # Calculate world points if needed
        if world_points is None:
            # Get lidar information
            lidar_pose = self.get_current_or_context('pose', depth_idx, context)
            lidar_points = self.get_current_or_context('point_cloud', depth_idx, context)
            world_points = lidar_pose * lidar_points
            # Calculate scene flow in world frame of reference
            if with_scene_flow:
                bwd_scene_flow = self.get_current_or_context('bwd_scene_flow', depth_idx, context)
                if bwd_scene_flow is not None:
                    bwd_scene_flow = lidar_pose * (lidar_points + bwd_scene_flow) - world_points
                fwd_scene_flow = self.get_current_or_context('fwd_scene_flow', depth_idx, context)
                if fwd_scene_flow is not None:
                    fwd_scene_flow = lidar_pose * (lidar_points + fwd_scene_flow) - world_points
        # Create camera
        camera = self.create_camera(camera_idx, context)
        image_shape = self.get_current_or_context('rgb', camera_idx, context).size[::-1]
        # Generate depth and scene flow maps
        depth, bwd_scene_flow, fwd_scene_flow = \
            generate_proj_maps(camera, world_points, image_shape, bwd_scene_flow, fwd_scene_flow)
        # Save depth map
        save_to_file(filename_depth, 'depth', depth)
        # Save scene flow
        if with_scene_flow:
            save_to_file(filename_bwd_scene_flow, 'scene_flow', bwd_scene_flow)
            save_to_file(filename_fwd_scene_flow, 'scene_flow', fwd_scene_flow)
        # Return depth and scene flow
        return depth, bwd_scene_flow, fwd_scene_flow

    def create_pointcache(self, filename, camera_idx, sample_idx, bbox3d):
        """Generate pointcache for a given bounding box

        Parameters
        ----------
        filename : str
            Filename
        camera_idx : int
            Camera index
        sample_idx : int
            Sample index
        bbox3d : dict
            Bounding box information

        Returns
        -------
        dict
            Pointcache information
        """
        # Get cache path
        filename_pkl = '{}/{}.pkl'.format(
            os.path.dirname(self.path), filename.format('cached_pointcache'))
        # Load and return if exists
        try:
            return pickle.load(open(filename_pkl, 'rb'))
        except:
            pass
        # Get pointcache if not provided
        pointcache = {'points': [], 'instance_id': []}
        cam_pose = self.get_current('pose', camera_idx)
        for k, b in enumerate(bbox3d):
            if 'point_cache' in b.attributes.keys():
                scene_idx, sample_idx_in_scene, _ = self.dataset.dataset_item_index[sample_idx]
                scene_dir = self.dataset.scenes[scene_idx].directory
                full_pcl = []
                for item in eval(b.attributes['point_cache']):
                    filename = os.path.join(self.path, scene_dir, 'point_cache', item['sha']) + '.npz'
                    pcl_raw = read_numpy(filename, key='data')
                    # Get points and normals
                    pcl = np.concatenate([pcl_raw['X'], pcl_raw['Y'], pcl_raw['Z']], 1)
                    nrm = np.concatenate([pcl_raw['NX'], pcl_raw['NY'], pcl_raw['NZ']], 1)
                    tvec, wxyz = item['pose']['translation'], item['pose']['rotation']
                    offset = Pose(tvec=np.float32([tvec['x'], tvec['y'], tvec['z']]),
                                  wxyz=np.float32([wxyz['qw'], wxyz['qx'], wxyz['qy'], wxyz['qz']]))
                    pcl = cam_pose * b.pose * offset * (pcl * item['size'])
                    # Concatenate points and normals
                    full_pcl.append(np.concatenate([pcl, nrm], 1))
                # Store information
                pointcache['points'].append(np.concatenate(full_pcl, 0))
                pointcache['instance_id'].append(b.instance_id)
        # Save pointcache
        os.makedirs(os.path.dirname(filename_pkl), exist_ok=True)
        with open(filename_pkl, "wb") as f:
            pickle.dump(pointcache, f)
        # Return pointcache
        return pointcache

    def get_keypoints(self, filename, rgb):
        """Get keypoints from filename."""
        keypoint_path = ('%s/keypoints/%s.txt.npz' % (os.path.dirname(self.path), filename)).format('rgb')
        keypoints = read_numpy(keypoint_path, key='data')
        keypoints_coord, keypoints_desc = keypoints[:, :2], keypoints[:, 2:]
        keypoints_coord[:, 0] *= rgb.size[0] / 320
        keypoints_coord[:, 1] *= rgb.size[1] / 240
        return keypoints_coord, keypoints_desc

    def get_current(self, key, sensor_idx, as_dict=False):
        """Return current timestep of a key from a sensor"""
        current = self.sample_dgp[self.bwd_context][sensor_idx][key]
        return current if not as_dict else {(0, sensor_idx): current}

    def get_backward(self, key, sensor_idx):
        """Return backward timesteps of a key from a sensor"""
        return [] if self.bwd_context == 0 else \
            [self.sample_dgp[i][sensor_idx][key] for i in range(0, self.bwd_context)]

    def get_forward(self, key, sensor_idx):
        """Return forward timesteps of a key from a sensor"""
        return [] if self.fwd_context == 0 else \
            [self.sample_dgp[i][sensor_idx][key]
             for i in range(self.bwd_context + 1,
                            self.bwd_context + self.fwd_context + 1)]

    def get_context(self, key, sensor_idx, as_dict=False):
        """Get both backward and forward contexts"""
        context = self.get_backward(key, sensor_idx) + self.get_forward(key, sensor_idx)
        if not as_dict:
            return context
        else:
            return {(key, sensor_idx): val for key, val in zip(self.context, context)}

    def get_current_or_context(self, key, sensor_idx, context=None, as_dict=False):
        """Return current or context information for a given key and sensor index"""
        if context is None:
            return self.get_current(key, sensor_idx, as_dict=as_dict)
        else:
            return self.get_context(key, sensor_idx, as_dict=as_dict)[context]

    def get_bbox3d(self, i):
        """Return dictionary with bounding box information"""
        bbox3d = self.get_current('bounding_box_3d', i)
        bbox3d = [b for b in bbox3d if b.num_points > 0]
        pose = self.get_current('pose', i)
        return bbox3d, {
            'pose': pose.matrix,
            'corners': np.stack([(pose * b).corners for b in bbox3d], 0),
            'class_id': np.stack([b.class_id for b in bbox3d], 0),
            'instance_id': np.stack([b.instance_id for b in bbox3d]),
        }

    def has_dgp_key(self, key, sensor_idx):
        """Returns True if the DGP sample contains a certain key"""
        return key in self.sample_dgp[self.bwd_context][sensor_idx].keys()

    def get_filename(self, sample_idx, datum_idx, context=0):
        """
        Returns the filename for an index, following DGP structure

        Parameters
        ----------
        sample_idx : int
            Sample index
        datum_idx : int
            Datum index
        context : int
            Context offset for the sample

        Returns
        -------
        filename : str
            Filename for the datum in that sample
        """
        scene_idx, sample_idx_in_scene, _ = self.dataset.dataset_item_index[sample_idx]
        scene_dir = self.dataset.scenes[scene_idx].directory
        filename = self.dataset.get_datum(
            scene_idx, sample_idx_in_scene + context, self.sensors[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def __len__(self):
        """Length of dataset"""
        return len(self.dataset)

    def save_pickle(self, idx):
        """Save pickle file with sample information"""
        split = self.path.split('/')
        path1 = '/'.join(split[:-2])
        path2 = f'{split[-2]}_{split[-1]}'
        folder = f'{path1}/{path2}_{self.split}'
        os.makedirs(folder, exist_ok=True)
        path = f'{folder}/%010d.pkl' % idx
        write_pickle(path, self.sample_dgp)

    def read_pickle(self, idx):
        """Read pickle file with sample information"""
        split = self.path.split('/')
        path1 = '/'.join(split[:-2])
        path2 = f'{split[-2]}_{split[-1]}'
        folder = f'{path1}/{path2}_{self.split}'
        os.makedirs(folder, exist_ok=True)
        path = f'{folder}/%010d.pkl' % idx
        return read_pickle(path)

    def __getitem__(self, idx, force_camera=None):
        """Get dataset sample given an index."""

        if self.fixed_idx is not None:
            idx = self.fixed_idx

        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]

        # Reorganize sensors to the right order
        sensor_names = [self.get_current('datum_name', i).lower() for i in range(len(self.sensors))]
        indexes = [sensor_names.index(v) for v in self.sensors]
        self.sample_dgp = [[s[idx] for idx in indexes] for s in self.sample_dgp]

        sample, idx = self.initialize_sample(idx)

        # Loop over all cameras
        for cam_idx, cam in enumerate(self.cameras if force_camera is None else [force_camera]):

            filename = self.get_filename(idx, cam_idx)
            time_cam = (0, cam_idx)
            update_dict(sample, 'filename', time_cam, filename)

            update_dict(sample, 'rgb', time_cam, self.get_current('rgb', cam_idx))
            if self.with_intrinsics:
                update_dict(sample, 'intrinsics', time_cam,
                            self.get_current('intrinsics', cam_idx))
            if self.with_pose:
                update_dict(sample, 'extrinsics', time_cam,
                            self.get_current('extrinsics', cam_idx).inverse().matrix)
                update_dict(sample, 'pose', time_cam,
                            self.get_current('pose', cam_idx).inverse().matrix)
            if self.with_depth:
                depth, bwd_scene_flow, fwd_scene_flow = self.create_proj_maps(
                    filename, cam_idx, self.depth_idx, self.depth_type,
                    with_scene_flow=self.with_scene_flow)
                update_dict(sample, 'depth', time_cam, depth)
            if self.with_optical_flow:
                if self.within_context(time_cam[0], 'bwd'):
                    update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] - 1, cam_idx),
                                       self.get_bwd_optical_flow(filename))
                if self.within_context(time_cam[0], 'fwd'):
                    update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] + 1, cam_idx),
                                   self.get_fwd_optical_flow(filename))

            if self.mask_rgb is not None:
                update_dict(sample, 'mask_rgb', time_cam,
                            read_image(os.path.join(self.mask_rgb, '%02d.png' % self.cameras[cam_idx])))

            if self.save_data:
                self.save_data_fn(filename, sample)

            # If context is returned
            if self.with_context:

                filename_context = {}
                for context in range(-self.bwd_context, 0):
                    filename_context[context] = self.get_filename(idx, cam_idx, context)
                for context in range(1, self.fwd_context + 1):
                    filename_context[context] = self.get_filename(idx, cam_idx, context)

                for time, filename in filename_context.items():
                    time_cam = (time, cam_idx)
                    update_dict(sample, 'filename', time_cam, filename)

                    sample['rgb'].update(self.get_context('rgb', cam_idx, as_dict=True))
                    if self.with_intrinsics_context:
                        sample['intrinsics'].update(self.get_current('intrinsics', cam_idx, as_dict=True))
                    if self.with_depth_context:
                        depth, bwd_scene_flow, fwd_scene_flow = self.create_proj_maps(
                            filename, cam_idx, self.depth_idx, self.depth_type,
                            with_scene_flow=self.with_scene_flow)
                        update_dict(sample, 'depth', time_cam, depth)
                    if self.with_pose_context:
                        sample['extrinsics'].update(
                            {key: val.inverse().matrix for key, val in
                             self.get_context('extrinsics', cam_idx, as_dict=True).items()})
                        sample['pose'].update(
                            {key: val.inverse().matrix for key, val in
                             self.get_context('pose', cam_idx, as_dict=True).items()})
                    if self.with_optical_flow_context:
                        if self.within_context(time_cam[0], 'bwd'):
                            update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] - 1, cam_idx),
                                               self.get_bwd_optical_flow(filename))
                        if self.within_context(time_cam[0], 'fwd'):
                            update_dict_nested(sample, 'optical_flow', time_cam, (time_cam[0] + 1, cam_idx),
                                               self.get_fwd_optical_flow(filename))

                    if self.mask_rgb is not None:
                        update_dict(sample, 'mask_rgb', time_cam,
                                    read_image(os.path.join(self.mask_rgb, '%02d.png' % self.cameras[cam_idx])))

        return self.post_process_sample(sample)

