

import numpy as np

from efm_datasets.dataloaders.BaseDataset import BaseDataset
from efm_datasets.dataloaders.utils.FolderTree import FolderTree
from efm_datasets.dataloaders.utils.misc import update_dict, invert_pose
from efm_datasets.utils.read import read_image


class ImageDataset(BaseDataset):
    """Image dataset class, for data composed of only image sequences.

    Parameters
    ----------
    single_camera : bool, optional
        True if dataset is composed of a single folder sequence, by default True
    extension : str, optional
        Image file extension, by default 'png'
    folders_start : str, optional
        Filter used to keep only folders with certain prefix, by default None
    mask_rgb : str, optional
        Path to image mask, by default None
    """
    def __init__(self, single_folder=False, extension='png', folders_start=None, mask_rgb=None, **kwargs):
        super().__init__(**kwargs, base_tag='folder')
        if self.split is None or self.split == '':
            self.split = ('', )
        self.rgb_tree = FolderTree(
            self.path, context=self.context, sub_folders=self.split,
            folders_start=folders_start, single_folder=single_folder, suffix='.' + extension)
        self.mask_rgb = mask_rgb

    def __len__(self):
        """Dataset length"""
        return len(self.rgb_tree)

    @staticmethod
    def get_rgb(filename):
        """Get image from filename."""
        return read_image(filename)

    def get_intrinsics(self):
        """Get hard-coded intrinsics."""
        return np.array([
            [914.00086535,   0.        , 925.5502823 ], 
            [  0.        , 917.07328597, 379.95179225],
            [  0.        ,   0.        ,   1.        ]
        ])

    def __getitem__(self, idx):
        """Get dataset sample given an index"""

        # Initialize sample
        sample, idx = self.initialize_sample(idx)

        # Loop over all requested cameras
        for cam_idx, cam in enumerate(self.cameras):
            filename = self.rgb_tree.get_item(idx)[0]
            time_cam = (0, cam_idx)
            update_dict(sample, 'filename', time_cam, 
                        filename[:-4].replace(self.path + '/', ''))
            if self.with_rgb:
                update_dict(sample, 'rgb', time_cam, 
                            self.get_rgb(filename))
            if self.with_intrinsics:
                update_dict(sample, 'intrinsics', time_cam, 
                            self.get_intrinsics())
            if self.mask_rgb is not None:
                mask = filename.split('/')[-2].split('_')[0]
                mask = read_image('%s/%s.png' % (self.mask_rgb, mask), mode='L')
                sample['mask_rgb'] = {time_cam: mask}
            self.add_dummy_data(sample, time_cam)

            # If includes context
            if self.with_context:
                # Get context filenames and loop over them
                filename_context = self.rgb_tree.get_context(idx)
                for time, filename in filename_context.items():
                    time_cam = (time, cam_idx)
                    # Add information to the sample dictionary based on label request
                    update_dict(sample, 'filename', time_cam, 
                                filename[:-4].replace(self.path + '/', ''))
                    update_dict(sample, 'rgb', time_cam,
                                self.get_rgb(filename))
                    if self.with_intrinsics_context:
                        update_dict(sample, 'intrinsics', time_cam,
                                    self.get_intrinsics(filename, sample['rgb'][time_cam]))
                    if self.with_depth_context:
                        update_dict(sample, 'depth', time_cam,
                                    self.get_depth(filename, sample['rgb'][time_cam]))
                    if self.with_pose_context:
                        update_dict(sample, 'pose', time_cam,
                                    self.get_pose(filename))
                    self.add_dummy_data_context(sample, time_cam)

        # Return post-processed sample
        return self.post_process_sample(sample)


