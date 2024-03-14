
import os
from glob import glob

import numpy as np
from efm_datasets.utils.data import make_list, flatten
from efm_datasets.utils.types import is_list


def any_ends_with(files, invalids):
    return [f for f in files if not any([f.endswith(i) for i in invalids])]


def any_has(files, invalids):
    return [f for f in files if not any([i in f for i in invalids])]


class FolderTree:
    def __init__(self, path, prefix='', suffix='', sub_folders=('',), deep=1, 
                 invalids_has=None, invalids_end=None,
                 start=None, finish=None, single_folder=False, nested=False, filter_nested=None,
                 keep_folders=None, remove_folders=None, folders_start=None, remove_files=None, stride=1, 
                 context=(), context_type='temporal'):
        """Folder tree data class, used to store and load data from multiple sequence folders with temporal context.

        Parameters
        ----------
        path : str
            Dataset folder
        prefix : str, optional
            Prefix used to determine which folders to use, by default ''
        suffix : str, optional
            Suffix used to determine which folders to use, by default ''
        sub_folders : tuple, optional
            Sub-folders inside sequence folders that should be considered, by default ('',)
        deep : int, optional
            How deep the relevant folders are inside each folder sequence, by default 1
        invalids_has : str, optional
            Pattern used to filter out invalid folders, by default None
        invalids_end : str, optional
            Pattern used to filter out invalid folders, by default None
        start : int, optional
            Starting point for each sequence, by default None
        finish : int, optional
            Ending point for each sequence, by default None
        single_folder : bool, optional
            True if dataset is composed of a single folder sequence, by default True
        nested : bool, optional
            True if sequence folders are nested recursively, by default False
        filter_nested : str, optional
            Pattern used to filter out nested sequence folder, by default None
        keep_folders : str, optional
            Pattern used to decide which folders are kept, by default None
        remove_folders : str, optional
            Pattern used to decide which folders are removed, by default None
        folders_start : str, optional
            Pattern used to decide which folders are used, by default None
        remove_files : str, optional
            Pattern used to decide which files are removed, by default None
        stride : int, optional
            Temporal context stride, by default 1
        context : tuple, optional
            Temporal context, by default ()
        context_type : str, optional
            Type of context used, by default 'temporal'
        """
        # Store context information
        self.context = list(context)
        if 0 not in self.context:
            self.context.append(0)
        self.num_context = 0 if len(self.context) == 0 else max(self.context) - min(self.context)
        self.with_context = self.num_context > 0
        self.min_context = 0 if not self.with_context else min(self.context)
        self.context_type = context_type
        self.single_slice = 'single' in context_type

        self.stride = stride
        self.pad_numbers = False

        # Initialize empty folder tree
        self.folder_tree = []

        # If we are providing a file list, treat each line as a scene
        if is_list(path):
            self.folder_tree = [[file] for file in path]
        # If we are providing a folder
        else:
            # Get folders
            string = '*' + '/*' * (deep - 1)
            folders = glob(os.path.join(path, string))
            folders.sort()

            # Remove and keep folders as needed
            if keep_folders is not None:
                folders = [f for f in folders if os.path.basename(f) in keep_folders]
            if remove_folders is not None:
                folders = [f for f in folders if os.path.basename(f) not in remove_folders]
            if folders_start is not None:
                folders = flatten(
                    [[f for f in folders if os.path.basename(f).startswith(str(start))] for start in folders_start])

            # If nesting, go one folder deeper in order to find the scenes
            if nested:
                upd_folders = []
                for folder in folders:
                    new_folders = glob(os.path.join(folder, '*'))
                    upd_folders.extend(new_folders)
                folders = upd_folders
                folders.sort()
                if filter_nested is not None:
                    folders = [f for f in folders if f.split('/')[-1] == filter_nested]

            if single_folder:
                # Use current folder as the only one
                files = [file for file in folders if file.endswith(suffix)]
                self.folder_tree.append(files)
            else:
                # Populate folder tree
                for folder in folders:
                    # For each sub-folder
                    for sub_folder in make_list(sub_folders):
                        # Get and sort files in each folder
                        files = glob(os.path.join(folder, sub_folder, '{}*{}'.format(prefix, suffix)))
                        if self.pad_numbers:
                            for i in range(len(files)):
                                pref, suf = files[i].split('/')[:-1], files[i].split('/')[-1]
                                num, ext = suf.split('.')
                                files[i] = '/'.join(pref) + ('/%010d' % int(num)) + '.' + ext
                        # if len(remove) > 0:
                        #     for rem in remove:
                        #         files = [file for file in files if rem not in file]
                        files.sort()
                        if start is not None:
                            files = files[start:]
                        if finish is not None:
                            files = files[:finish]
                        if remove_files is not None:
                            for remove in make_list(remove_files):
                                files = [f for f in files if remove not in f]
                        if self.pad_numbers:
                            for i in range(len(files)):
                                pref, suf = files[i].split('/')[:-1], files[i].split('/')[-1]
                                num, ext = suf.split('.')
                                files[i] = '/'.join(pref) + ('/%d' % int(num)) + '.' + ext
                        if self.stride > 1:
                            files = files[::self.stride]

                        if invalids_has is not None:
                            files = any_has(files, invalids_has)
                        if invalids_end is not None:
                            files = any_ends_with(files, invalids_end)

                        # Only store if there are more images than context
                        if len(files) > self.num_context:
                            self.folder_tree.append(files)

        # Prepare additional structures
        self.slices = self.total = None
        self.prepare()

    def __len__(self):
        """Dataset length"""
        if self.single_slice:
            return len(self.slices) - 1
        else:
            return self.total

    def prepare(self):
        """Prepare folder tree and additional structures"""
        # Get size of each folder
        self.slices = [len(folder) for folder in self.folder_tree]
        # Compensate for context size
        if self.with_context:
            self.slices = [s - self.num_context for s in self.slices]
        # Create cumulative size and get total
        self.slices = [0] + list(np.cumsum(self.slices))
        self.total = self.slices[-1]

    def get_idxs(self, idx):
        """Get folder and file indexes given dataset index"""
        if self.single_slice:
            return idx, 0
        else:
            idx1 = np.searchsorted(self.slices, idx, side='right') - 1
            idx2 = idx - self.slices[idx1]
            return idx1, idx2

    def get_context_idxs(self, idx):
        """Get context indexes given dataset index"""
        idx1, idx2 = self.get_idxs(idx)
        return idx1, list(range(0, idx2)) + list(range(idx2 + 1, self.slices[idx1 + 1]))

    def get_proximity(self, idx1, offset):
        """Get folder and file indexes given dataset index"""
        return self.folder_tree[idx1][offset - self.min_context]

    def get_item(self, idx, return_loc=False):
        """Return filename item given index"""
        idx1, idx2 = self.get_idxs(idx)
        item = {0: self.folder_tree[idx1][idx2 - self.min_context]}
        if return_loc:
            return item, idx2 - self.min_context
        else:
            return item

    def get_context(self, idx, remove_target=True, context=None):
        """Return forward context given index."""
        idx1, idx2 = self.get_idxs(idx)
        if context is None: context = self.context
        context = {ctx: self.folder_tree[idx1][idx2 - self.min_context + ctx] for ctx in context}
        if remove_target:
            for tgt in [0, (0, 0)]:
                if tgt in list(context.keys()):
                    context.pop(tgt)
        return context

    def get_random(self, idx, qty):
        """Get random contexts given index"""
        idx1, idx2 = self.get_idxs(idx)

        n, m = len(self.folder_tree[idx1]), self.slices[idx1]
        rnd = np.random.permutation(n)
        rnd = [i for i in rnd if i != idx2]

        idxs = [idx]
        for i in range(qty - 1):
            idxs.append(rnd[i] + m)

        return idxs

    def get_full_context(self, idx):
        """Get entire sequence context"""
        idx1, idx2 = self.get_idxs(idx)
        scene = self.folder_tree[idx1]
        bwd, fwd = idx2, len(scene) - idx2
        context = list(range(bwd, fwd))
        return self.get_context(idx, context=context)
