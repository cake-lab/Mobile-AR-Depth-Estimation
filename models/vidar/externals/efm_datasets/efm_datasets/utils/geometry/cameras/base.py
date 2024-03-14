from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from efm_datasets.utils.geometry.camera_utils import invert_intrinsics, scale_intrinsics
from efm_datasets.utils.geometry.pose import Pose
from efm_datasets.utils.geometry.pose_utils import invert_pose
from efm_datasets.utils.tensor import pixel_grid, same_shape, cat_channel_ones, norm_pixel_grid, interpolate, interleave
from efm_datasets.utils.types import is_tensor, is_seq, is_tuple
from einops import rearrange


class CameraBase(nn.Module, ABC):
    def __init__(self, hw, Twc=None, Tcw=None):
        super().__init__()
        assert Twc is None or Tcw is None

        # Pose

        if Twc is None and Tcw is None:
            self._Twc = torch.eye(
                4, dtype=self._K.dtype, device=self._K.device).unsqueeze(0).repeat(self._K.shape[0], 1, 1)
        else:
            self._Twc = invert_pose(Tcw) if Tcw is not None else Twc
        if is_tensor(self._Twc):
            self._Twc = Pose(self._Twc)

        # Resolution

        self._hw = hw
        if is_tensor(self._hw):
            self._hw = self._hw.shape[-2:]

    def __getitem__(self, idx):
        """Return batch-wise pose"""
        if is_seq(idx):
            return type(self).from_list([self.__getitem__(i) for i in idx])
        else:
            if not is_tensor(idx):
                idx = [idx]
            return type(self)(
                K=self._K[idx],
                Twc=self._Twc[idx] if self._Twc is not None else None,
                hw=self._hw,
            )

    def __len__(self):
        """Return length as intrinsics batch"""
        return self._K.shape[0]

    def __eq__(self, cam):
        if not isinstance(cam, type(self)):
            return False
        if self._hw[0] != cam.hw[0] or self._hw[1] != cam.hw[1]:
            return False
        if not torch.allclose(self._K, cam.K):
            return False
        if not torch.allclose(self._Twc.T, cam.Twc.T):
            return False
        return True

    def clone(self):
        return type(self)(
            K=self.K.clone(),
            Twc=self.Twc.clone(),
            hw=[v for v in self._hw],
        )

    @property
    def pose(self):
        return self._Twc.T

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, K):
        self._K = K

    @property
    def batch_size(self):
        return self._Twc.T.shape[0]

    @property
    def b(self):
        return self._Twc.T.shape[0]

    @property
    def bhw(self):
        return self.b, self.hw

    @property
    def bdhw(self):
        return self.b, self.device, self.hw

    @property
    def hw(self):
        return self._hw

    @hw.setter
    def hw(self, hw):
        self._hw = hw

    @property
    def wh(self):
        return self._hw[::-1]

    @property
    def n_pixels(self):
        return self._hw[0] * self._hw[1]

    @property
    def Tcw(self):
        return None if self._Twc is None else self._Twc.inverse()

    @Tcw.setter
    def Tcw(self, Tcw):
        self._Twc = Tcw.inverse()

    @property
    def Twc(self):
        return self._Twc

    @Twc.setter
    def Twc(self, Twc):
        self._Twc = Twc

    @property
    def dtype(self):
        return self._K.dtype

    @property
    def device(self):
        return self._K.device

    def detach_pose(self):
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def detach_K(self):
        return type(self)(K=self._K.detach(), hw=self._hw, Twc=self._Twc)

    def detach(self):
        return type(self)(K=self._K.detach(), hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def inverted_pose(self):
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.inverse() if self._Twc is not None else None)

    def no_translation(self):
        Twc = self.pose.clone()
        Twc[:, :-1, -1] = 0
        return type(self)(K=self._K, hw=self._hw, Twc=Twc)

    def no_pose(self):
        return type(self)(K=self._K, hw=self._hw)

    def interpolate(self, rgb):
        if rgb.dim() == 5:
            rgb = rearrange(rgb, 'b n c h w -> (b n) c h w')
        return interpolate(rgb, scale_factor=None, size=self.hw, mode='bilinear')

    def interleave_K(self, b):
        return type(self)(
            K=interleave(self._K, b),
            Twc=self._Twc,
            hw=self._hw,
        )

    def interleave_Twc(self, b):
        return type(self)(
            K=self._K,
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def interleave(self, b):
        return type(self)(
            K=interleave(self._K, b),
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def repeat_bidir(self):
        return type(self)(
            K=self._K.repeat(2, 1, 1),
            Twc=torch.cat([self._Twc.T, self.Tcw.T], 0),
            hw=self.hw,
        )

    def Pwc(self, from_world=True):
        return self._K[:, :3] if not from_world or self._Twc is None else \
            torch.matmul(self._K, self._Twc.T)[:, :3]

    def to_world(self, points):
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self.Tcw is None else self.Tcw * points

    def from_world(self, points):
        if points.dim() > 3:
            shape = points.shape
            points = points.reshape(points.shape[0], 3, -1)
        else:
            shape = None
        local_points = points if self._Twc is None else \
            torch.matmul(self._Twc.T, cat_channel_ones(points, 1))[:, :3]
        return local_points if shape is None else local_points.view(shape)
    
    def from_world2(self, points):
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self._Twc is None else \
            torch.matmul(self._Twc.T[:, :3, :3], points[:, :3]) + self._Twc.T[:, :3, 3:]

    def to(self, *args, **kwargs):
        self._K = self._K.to(*args, **kwargs)
        if self._Twc is not None:
            self._Twc = self._Twc.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        return self.to('cuda')

    def relative_to(self, cam):
        return type(self)(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc.inverse())

    def global_from(self, cam):
        return type(self)(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc)

    def pixel_grid(self, shake=False):
        return pixel_grid(
            b=self.batch_size, hw=self.hw, with_ones=True,
            shake=shake, device=self.device).view(self.batch_size, 3, -1)

    def reconstruct_depth_map(self, depth, to_world=False, grid=None, scene_flow=None, world_scene_flow=None):
        if depth is None:
            return None
        b, _, h, w = depth.shape
        if grid is None:
            grid = pixel_grid(depth, with_ones=True, device=depth.device).view(b, 3, -1)
        points = self.lift(grid) * depth.view(depth.shape[0], 1, -1)
        if scene_flow is not None:
            points = points + scene_flow.view(b, 3, -1)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
            if world_scene_flow is not None:
                points = points + world_scene_flow.view(b, 3, -1)
        return points.view(b, 3, h, w)

    def reconstruct_euclidean(self, depth, to_world=False, grid=None, scene_flow=None, world_scene_flow=None):
        if depth is None:
            return None
        b, _, h, w = depth.shape
        rays = self.get_viewdirs(normalize=True, to_world=False, grid=grid).view(b, 3, -1)
        points = rays * depth.view(depth.shape[0], 1, -1)
        if scene_flow is not None:
            points = points + scene_flow.view(b, 3, -1)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
            if world_scene_flow is not None:
                points = points + world_scene_flow.view(b, 3, -1)
        return points.view(b, 3, h, w)

    def reconstruct_volume(self, depth, euclidean=False, **kwargs):
        if euclidean:
            return torch.stack([self.reconstruct_euclidean(depth[:, :, i], **kwargs)
                                for i in range(depth.shape[2])], 2)
        else:
            return torch.stack([self.reconstruct_depth_map(depth[:, :, i], **kwargs)
                                for i in range(depth.shape[2])], 2)

    # def reconstruct_cost_volume(self, volume, to_world=True, flatten=True):
    #     c, d, h, w = volume.shape
    #     grid = pixel_grid((h, w), with_ones=True, device=volume.device).view(3, -1).repeat(1, d)
    #     points = torch.stack([
    #         (volume.view(c, -1) * torch.matmul(invK[:3, :3].unsqueeze(0), grid)).view(3, d * h * w)
    #         for invK in self.invK], 0)
    #     if to_world and self.Tcw is not None:
    #         points = self.Tcw * points
    #     if flatten:
    #         return points.view(-1, 3, d, h * w).permute(0, 2, 1, 3)
    #     else:
    #         return points.view(-1, 3, d, h, w)

    def project_points(self, points, from_world=True, normalize=True,
                       return_z=False, return_e=False, flag_invalid=True):

        is_depth_map = points.dim() == 4
        hw = self._hw if not is_depth_map else points.shape[-2:]
        return_depth = return_z or return_e

        if is_depth_map:
            points = points.reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        coords, depth = self.unlift(points, from_world=from_world, euclidean=return_e)

        if not is_depth_map:
            if normalize:
                coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
                if flag_invalid:
                    invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                              (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth < 0)
                    coords[invalid.unsqueeze(1).repeat(1, 2, 1)] = -2
            if return_depth:
                return coords.permute(0, 2, 1), depth
            else:
                return coords.permute(0, 2, 1)

        coords = coords.view(b, 2, *hw)
        depth = depth.view(b, 1, *hw)

        if normalize:
            coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
            if flag_invalid:
                invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                          (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth[:, 0] < 0)
                coords[invalid.unsqueeze(1).repeat(1, 2, 1, 1)] = -2

        if return_depth:
            return coords.permute(0, 2, 3, 1), depth
        else:
            return coords.permute(0, 2, 3, 1)

    # def project_cost_volume(self, points, from_world=True, normalize=True):
    #
    #     if points.dim() == 4:
    #         points = points.permute(0, 2, 1, 3).reshape(points.shape[0], 3, -1)
    #     b, _, n = points.shape
    #
    #     points = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))
    #
    #     coords = points[:, :2] / (points[:, 2].unsqueeze(1) + 1e-7)
    #     coords = coords.view(b, 2, -1, *self._hw).permute(0, 2, 3, 4, 1)
    #
    #     if normalize:
    #         coords[..., 0] /= self._hw[1] - 1
    #         coords[..., 1] /= self._hw[0] - 1
    #         return 2 * coords - 1
    #     else:
    #         return coords

    def create_radial_volume(self, bins, to_world=True):
        ones = torch.ones((1, *self.hw), device=self.device)
        volume = torch.stack([depth * ones for depth in bins], 1).unsqueeze(0)
        return self.reconstruct_volume(volume, to_world=to_world)

    def project_volume(self, volume, from_world=True):
        b, c, d, h, w = volume.shape
        return self.project_points(volume.view(b, c, -1), from_world=from_world).view(b, d, h, w, 2)

    # def coords_from_cost_volume(self, volume, ref_cam=None):
    #     if ref_cam is None:
    #         return self.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=False), from_world=True)
    #     else:
    #         return ref_cam.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=True), from_world=True)
    #
    # def coords_from_depth(self, depth, ctx_cam=None, scene_flow=None, world_scene_flow=None):
    #     if ctx_cam is None:
    #         return self.project_points(self.reconstruct_depth_map(
    #             depth, to_world=False, scene_flow=scene_flow, world_scene_flow=world_scene_flow), from_world=True)
    #     else:
    #         return ctx_cam.project_points(self.reconstruct_depth_map(
    #             depth, to_world=True, scene_flow=scene_flow, world_scene_flow=world_scene_flow), from_world=True)

    def z2e(self, z_depth):
        points = self.reconstruct_depth_map(z_depth, to_world=False)
        return self.project_points(points, from_world=False, return_e=True)[1]

    def e2z(self, e_depth):
        points = self.reconstruct_euclidean(e_depth, to_world=False)
        return self.project_points(points, from_world=False, return_z=True)[1]

    def control(self, draw, tvel=0.2, rvel=0.1):
        change = False
        if draw.UP:
            self.Twc.translateForward(tvel)
            change = True
        if draw.DOWN:
            self.Twc.translateBackward(tvel)
            change = True
        if draw.LEFT:
            self.Twc.translateLeft(tvel)
            change = True
        if draw.RIGHT:
            self.Twc.translateRight(tvel)
            change = True
        if draw.PGUP:
            self.Twc.translateUp(tvel)
            change = True
        if draw.PGDOWN:
            self.Twc.translateDown(tvel)
            change = True
        if draw.KEY_A:
            self.Twc.rotateYaw(-rvel)
            change = True
        if draw.KEY_D:
            self.Twc.rotateYaw(+rvel)
            change = True
        if draw.KEY_W:
            self.Twc.rotatePitch(+rvel)
            change = True
        if draw.KEY_S:
            self.Twc.rotatePitch(-rvel)
            change = True
        if draw.KEY_Q:
            self.Twc.rotateRoll(-rvel)
            change = True
        if draw.KEY_E:
            self.Twc.rotateRoll(+rvel)
            change = True
        return change
