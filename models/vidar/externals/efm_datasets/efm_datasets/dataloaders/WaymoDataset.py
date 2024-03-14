import os
import numpy as np
import torch

from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.dataloaders.utils.FolderTree import FolderTree
from efm_datasets.dataloaders.utils.misc import update_dict
from efm_datasets.utils.read import read_image, read_numpy
from efm_datasets.utils.write import write_npz

from efm_datasets.utils.geometry.camera import Camera


class WaymoDataset(BaseDataset):
    """Waymo dataset class. 
    
    https://waymo.com/open/

    Parameters
    ----------
    save_depth_maps : bool, optional
        True if depth maps are saved for later use, by default False
    resolution : tuple, optional
        Requested sample resolution, by default None
    focal_ratio : float, optional
        Focal length resizing (for changes in aspect ratio), by default 1.0
    """
    def __init__(self, save_depth_maps=False, resolution=None, focal_ratio=1.0, **kwargs):
        super().__init__(**kwargs, base_tag='waymo')

        self.rgb_tree = {}
        for cam in self.cameras:
            files = open(os.path.join(self.path, f'ImageSets/{self.split}.txt'), 'r')
            files = files.readlines()
            files = ['%s/%s/image_%d/%07d.png' % (self.path, self.split, cam, int(file)) for file in files]
            self.rgb_tree[cam] = FolderTree(
                files, context=self.context, sub_folders=self.split,
                single_folder=False, suffix='.png')
        self.save_depth_maps = save_depth_maps
        self.multi_resolution = resolution is not None and len(resolution) == 4
        self.resolution = None if resolution is None else \
            tuple(resolution) if not self.multi_resolution else \
            [tuple(resolution[:2]), tuple(resolution[2:])]
        self.focal_ratio = focal_ratio

    def __len__(self):
        """Dataset length"""
        return len(self.rgb_tree[self.cameras[0]])

    def get_resolution(self, filename):
        """Get resolution from filename."""
        if self.resolution is not None:
            if not self.multi_resolution:
                resolution = self.resolution
            else:
                if '_0/' in filename or '_1/' in filename or '_2/' in filename:
                    resolution = self.resolution[0]
                else:
                    resolution = self.resolution[1]
        return resolution

    def get_rgb(self, filename):
        """Get image from filename."""
        if self.resolution is not None:
            resolution = self.get_resolution(filename)
            filename = filename.replace('image', 'image_%d_%d' % resolution).replace('.png', '.jpg')
        return read_image(filename)

    def get_intrinsics(self, filename):
        """Get intrinsics from filename."""
        if self.resolution is not None:
            resolution = self.get_resolution(filename)
            filename = filename.replace('image', 'intrinsics_%d_%d' % resolution).replace('.png', '.npy')
            intrinsics = read_numpy(filename)
            if self.focal_ratio != 1.0:
                intrinsics[0, 0] *= self.focal_ratio
                intrinsics[1, 1] *= self.focal_ratio
            return intrinsics

        split = filename.split('/')
        cam = int(split[-2].split('_')[-1])
        filename = os.path.join('/'.join(split[:-2]), 'calib', split[-1]).replace('.png', '.txt')
        with open(filename) as f:
            lines = f.read().splitlines()
        intrinsics = np.array([float(val) for val in lines[cam].split(' ')[1:]]).reshape(3, 4)[:, :-1]
        return intrinsics

    def get_pose(self, filename):
        """Get pose from filename."""
        split = filename.split('/')
        abs_filename = os.path.join('/'.join(split[:-2]), 'pose', split[-1]).replace('.png', '.txt')
        abs_pose = np.loadtxt(abs_filename, dtype=np.float32)

        split = filename.split('/')
        cam = int(split[-2].split('_')[-1])
        filename = os.path.join('/'.join(split[:-2]), 'calib', split[-1]).replace('.png', '.txt')
        with open(filename) as f:
            lines = f.read().splitlines()
        pose = np.array([float(val) for val in lines[cam + 6].split(' ')[1:]]).reshape(3, 4)
        pose = np.vstack([pose, np.array([[0.0, 0.0, 0.0, 1.0]])])
        pose = pose @ abs_pose

        return pose

    def get_depth(self, rgb, filename):
        """Get depth from filename."""
        if self.resolution is not None:
            resolution = self.get_resolution(filename)
            filename = filename.replace('image', 'depth_%d_%d' % resolution).replace('.png', '.npz')
            return read_numpy(filename)['depth']
        if not self.save_depth_maps:
            split = filename.split('/')
            cam = int(split[-2].split('_')[-1])
            filename_depth = os.path.join('/'.join(split[:-2]), 'depth_%d' % cam, split[-1]).replace('.png', '.npz')
            return read_numpy(filename_depth)['depth']

        shape = rgb.size[::-1]

        split = filename.split('/')
        cam = int(split[-2].split('_')[-1])
        K_filename = os.path.join('/'.join(split[:-2]), 'calib', split[-1]).replace('.png', '.txt')
        with open(K_filename) as f:
            lines = f.read().splitlines()
        intrinsics = np.array([float(val) for val in lines[cam].split(' ')[1:]], dtype=np.float32).reshape(3, 4)[:, :-1]

        split = filename.split('/')
        cam = int(split[-2].split('_')[-1])
        pose_filename = os.path.join('/'.join(split[:-2]), 'calib', split[-1]).replace('.png', '.txt')
        with open(pose_filename) as f:
            lines = f.read().splitlines()
        pose = np.array([float(val) for val in lines[cam + 6].split(' ')[1:]]).reshape(3, 4)
        pose = np.vstack([pose, np.array([[0.0, 0.0, 0.0, 1.0]])])

        split = filename.split('/')
        filename_velodyne = os.path.join('/'.join(split[:-2]), 'velodyne', split[-1]).replace('.png', '.bin')
        points = np.fromfile(filename_velodyne, dtype=np.float32).reshape(-1, 3)

        points = torch.tensor(points[:, :3]).permute(1, 0).unsqueeze(0).float()

        cam = Camera(
            K=torch.tensor(intrinsics).unsqueeze(0).float(),
            Twc=torch.tensor(pose).unsqueeze(0).float(),
            hw=shape,
        )

        uv, z_c = cam.project_points(points, from_world=True, return_z=True, normalize=False)
        uv, z_c = uv.numpy().astype(int)[0], z_c.numpy()[0]

        H, W = shape
        proj_depth = 0 * np.ones((H, W), dtype=np.float32)
        in_view = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W) & (uv[:, 1] < H) & (z_c > 0)
        uv, z_c = uv[in_view], z_c[in_view]
        proj_depth[uv[:, 1], uv[:, 0]] = z_c

        proj_depth = np.expand_dims(proj_depth, 2)

        split = filename.split('/')
        cam = int(split[-2].split('_')[-1])
        filename_depth = os.path.join('/'.join(split[:-2]), 'depth_%d' % cam, split[-1]).replace('.png', '.npz')
        write_npz(filename_depth, {'depth': proj_depth})
        return proj_depth

    def __getitem__(self, idx):
        """Get dataset sample given an index."""

        # Initialize base sample
        sample, idx = self.initialize_sample(idx)

        # Loop over all requested cameras
        for cam_idx, cam in enumerate(self.cameras):

            # Get target filename and prepare key tuple
            filename = self.rgb_tree[cam].get_item(idx)[0]
            time_cam = (0, cam_idx)
            update_dict(sample, 'filename', time_cam, filename)

            # Add information to the sample dictionary based on label request
            update_dict(sample, 'rgb', time_cam, self.get_rgb(filename))
            if self.with_intrinsics:
                update_dict(sample, 'intrinsics', time_cam,
                            self.get_intrinsics(filename))
            if self.with_depth:
                update_dict(sample, 'depth', time_cam,
                            self.get_depth(sample['rgb'][time_cam], filename))
            if self.with_pose:
                update_dict(sample, 'pose', time_cam,
                            self.get_pose(filename))

            # If includes context
            if self.with_context:
                # Get context filenames and loop over them
                filename_context = self.rgb_tree.get_context(idx)
                for time, filename in filename_context.items():
                    # Get context filename and prepare key tuple
                    time_cam = (time, cam_idx)
                    update_dict(sample, 'filename', time_cam, filename)
                    # Add information to the sample dictionary based on label request
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

        # Return post-processed sample
        return self.post_process_sample(sample)


