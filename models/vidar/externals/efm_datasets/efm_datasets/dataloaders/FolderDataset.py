import numpy as np

from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.dataloaders.utils.FolderTree import FolderTree
from efm_datasets.dataloaders.utils.misc import update_dict, invert_pose
from efm_datasets.utils.read import read_image, read_depth


class FolderDataset(BaseDataset):
    """Folder dataset, for generic data in a folder structure. 

    Parameters
    ----------
    single_camera : bool, optional
        True if dataset is composed of a single folder sequence, by default True
    rgb_extension : str, optional
        Image file extension, by default 'png'
    rgb_folder : str, optional
        Image folder, by default 'rgb'
    depth_folder : str, optional
        Depth folder, by default 'depth'
    pose_folder : str, optional
        Pose folder, by default 'pose'
    intrinsics_folder : str, optional
        Intrinsics folder, by default 'intrinsics'
    """
    def __init__(self, single_camera=True, rgb_extension='png', 
                 rgb_folder='rgb', depth_folder='depth', 
                 pose_folder='pose', intrinsics_folder='intrinsics', **kwargs):
        super().__init__(**kwargs, base_tag='folder')
        self.single_camera = single_camera
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.pose_folder = pose_folder
        self.intrinsics_folder = intrinsics_folder

        self.rgb_extension = '.png'
        self.depth_extension = '.png'

        tree_path = f'{self.rgb_folder}' if self.single_camera else \
                    f'{self.rgb_folder}/{self.cameras[0]}'
        self.rgb_tree = FolderTree(
            self.path, context=self.context, sub_folders=tree_path,
            suffix='.' + rgb_extension)

    def __len__(self):
        """Dataset length"""
        return len(self.rgb_tree)

    @staticmethod
    def get_rgb(filename):
        """Get image from filename"""
        return read_image(filename)

    def get_intrinsics(self, filename):
        """Get intrinsics from filename"""
        filename = filename.replace(self.rgb_folder, self.intrinsics_folder)[:-4] + '.npy'
        return np.load(filename)[:3, :3]

    def get_pose(self, filename):
        """Get pose from filename"""
        filename = filename.replace(self.rgb_folder, self.pose_folder)[:-4] + '.npy'
        return np.load(filename)

    def get_depth(self, filename):
        """Get depth from filename"""
        filename = filename.replace(self.rgb_folder, self.depth_folder)[:-4] + self.depth_extension
        return read_depth(filename)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""

        # Initialize sample
        sample, idx = self.initialize_sample(idx)

        # Loop over all requested cameras
        for cam_idx, cam in enumerate(self.cameras):
            # Get target filename and prepare key tuple
            filename = self.rgb_tree.get_item(idx)[0]
            if not self.single_camera:
                filename = filename.replace(self.cameras[0], self.cameras[cam_idx])
            time_cam = (0, cam_idx)
            # Add information to the sample dictionary based on label request
            update_dict(sample, 'filename', time_cam, 
                        filename[:-4].replace(self.path + '/', ''))
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
            self.add_dummy_data(sample, time_cam)

            # If includes context
            if self.with_context:
                # Get context filenames and loop over them
                filename_context = self.rgb_tree.get_context(idx)
                for time, filename in filename_context.items():
                    if not self.single_camera:  
                        filename = filename.replace(self.cameras[0], self.cameras[cam_idx])
                    time_cam = (time, cam_idx)
                    # Add information to the sample dictionary based on label request
                    update_dict(sample, 'filename', 
                                time_cam, filename[:-4].replace(self.path + '/', ''))
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
                    self.add_dummy_data_context(sample, time_cam)

        # Return post-processed sample
        return self.post_process_sample(sample)



