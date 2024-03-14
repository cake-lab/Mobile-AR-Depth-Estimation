

from efm_datasets.utils.geometry.cameras.pinhole import CameraPinhole
from efm_datasets.utils.geometry.cameras.ucm import CameraUCM


def Camera(K, hw, Twc=None, Tcw=None, geometry='pinhole'):
    if geometry == 'pinhole':
        return CameraPinhole(K, hw, Twc, Tcw)
    elif geometry == 'ucm':
        return CameraUCM(K, hw, Twc, Tcw)
    else:
        raise ValueError('Invalid camera geometry')