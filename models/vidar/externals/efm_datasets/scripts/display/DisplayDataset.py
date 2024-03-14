
import numpy as np
import torch
from camviz import BBox3D
from camviz import Camera as CameraCV
from camviz import Draw

from efm_datasets.utils.geometry.cameras.pinhole import CameraPinhole as Camera
from efm_datasets.utils.geometry.pose import Pose
from efm_datasets.utils.data import make_batch, fold_batch, get_from_dict, interleave_dict, modrem
from efm_datasets.utils.depth import calculate_normals
from efm_datasets.utils.viz import viz_depth, viz_optical_flow, viz_semantic, viz_normals


def change_key(data, c, n):
    steps = sorted(set([key[0] for key in data.keys()]))
    return steps[(steps.index(c) + n) % len(steps)]


class DisplayDataset:
    def __init__(self, dataset, virtual_pose=None):

        self.idx = 0
        self.tgt = (0, 0)
        self.dataset = dataset

        self.tasks = ['rgb', 'depth', 'normals', 'semantic', 'optical_flow', 'optical_flow']
        self.cam_colors = ['red', 'blu', 'gre', 'yel', 'mag', 'cya'] * 100
        self.offset = [None, None, None, None, -1, 1]

        data, wh, keys, offsets, cams, num_cams, points, points_normals, actions, language = self.process()

        self.draw = Draw((wh[0] * 4, wh[1] * 3), width=2100)
        self.draw.add2DimageGrid('img', (0.0, 0.0, 0.5, 1.0), n=(max(3, num_cams // 2), 2), res=wh)
        self.draw.add3Dworld('wld', (0.5, 0.0, 1.0, 1.0),
                             pose=cams[self.tgt].Tcw.T[0] if virtual_pose is None else virtual_pose,
                             enable_blending=False)

        self.draw.addTexture('img', n=num_cams)
        self.draw.addBuffer3f('pts', 1000000, n=num_cams)
        self.draw.addBuffer3f('clr', 1000000, n=num_cams)
        self.draw.addBuffer3f('pts_nrm', 1000000, n=num_cams)
        self.draw.addBuffer3f('clr_nrm', 1000000, n=num_cams)

        with_bbox3d = 'bbox3d' in data
        if with_bbox3d:
            bbox3d_corners = [[BBox3D(b) for b in bb] for bb in data['bbox3d']['corners']]

        with_pointcache = 'pointcache' in data
        if with_pointcache:
            pointcache = np.concatenate([np.concatenate(pp, 0) for pp in data['pointcache']['points']], 0)
            self.draw.addBufferf('pointcache', pointcache[:, :3])

    def process(self):

        data = self.dataset[self.idx]

        data = make_batch(data)
        data = fold_batch(data)

        rgb = data['rgb']
        intrinsics = get_from_dict(data, 'intrinsics')
        depth = get_from_dict(data, 'depth')
        pose = get_from_dict(data, 'pose')

        actions = get_from_dict(data, 'actions')
        language = get_from_dict(data, 'language')

        pose = Pose.from_dict(pose, to_global=True, zero_origin=False, broken=True)
        cams = Camera.from_dict(intrinsics, rgb, pose, broken=True)
        num_cams = len(set([key[1] for key in cams.keys()]))
        wh = rgb[self.tgt].shape[-2:][::-1]

        points = {}
        for key, val in cams.items():
            points[key] = cams[key].reconstruct_depth_map(
                depth[key], to_world=True)

        if depth is not None:
            data['normals'] = {key: calculate_normals(depth[key], cams[key], to_world=True) for key in depth.keys()}
            points_normals = {key: cams[key].reconstruct_depth_map(
                depth[key], to_world=True, world_scene_flow=data['normals'][key] / 5) for key in depth.keys()}
            points_normals = interleave_dict(points, points_normals)
        else:
            points_normals = None

        idx = [i for i in range(len(self.tasks)) if self.tasks[i] in data.keys()]
        keys, offsets = [self.tasks[i] for i in idx], [self.offset[i] for i in idx]

        return data, wh, keys, offsets, cams, num_cams, points, points_normals, actions, language

    def loop(self):

        data, wh, keys, offsets, cams, num_cams, points, points_normals, actions, language = self.process()
        camcv = {key: CameraCV.from_vidar(val, b=0, scale=0.1) for key, val in cams.items()}

        zeros3 = torch.zeros((wh[1], wh[0], 3))
        zeros4 = torch.zeros((wh[1] * wh[0], 4))

        t, k = 0, 0
        key = keys[k]
        change = True
        color = True
        show_normals = False

        while self.draw.input():
            if self.draw.SPACE:
                color = not color
                change = True
            if self.draw.RIGHT:
                change = True
                k = (k + 1) % len(keys)
                key = keys[k]
            if self.draw.LEFT:
                change = True
                k = (k - 1) % len(keys)
                key = keys[k]
            if self.draw.UP:
                change = True
                t = change_key(data[key], t, 1)
            if self.draw.DOWN:
                change = True
                t = change_key(data[key], t, -1)
            if self.draw.KEY_A and self.idx < len(self.dataset) - 1:
                change = True
                self.idx += 1
                data, wh, keys, offsets, cams, num_cams, points, points_normals, actions, language = self.process()
            if self.draw.KEY_Z and self.idx > 0:
                change = True
                self.idx -= 1
                data, wh, keys, offsets, cams, num_cams, points, points_normals, actions, language = self.process()
            if self.draw.KEY_X and points_normals is not None:
                show_normals = not show_normals

            if change:
                change = False
                camcv = {key: CameraCV.from_vidar(val, b=0, scale=0.1) for key, val in cams.items()}
                for i in range(num_cams):
                    img = data[key][(t, i)]
                    if key == 'rgb':
                        img = img[0]
                        self.draw.updTexture('img%d' % i, img)
                        self.draw.updBufferf('clr%d' % i, img)
                    elif key == 'depth':
                        img = viz_depth(img[0], filter_zeros=True)
                        self.draw.updTexture('img%d' % i, img)
                        self.draw.updBufferf('clr%d' % i, img.reshape(-1, 3))
                    elif key == 'normals':
                        img = viz_normals(img[0])
                        self.draw.updTexture('img%d' % i, img)
                        self.draw.updBufferf('clr%d' % i, img.reshape(-1, 3))
                    elif key == 'optical_flow':
                        if (t + offsets[k], i) in img.keys():
                            img = img[(t + offsets[k], i)]
                            img = viz_optical_flow(img[0])
                            self.draw.updTexture('img%d' % i, img)
                            self.draw.updBufferf('clr%d' % i, img.reshape(-1, 3))
                        else:
                            self.draw.updTexture('img%d' % i, zeros3)
                            self.draw.updBufferf('clr%d' % i, zeros4)
                    elif key == 'semantic':
                        img = viz_semantic(img[0], self.ontology)
                        self.draw.updTexture('img%d' % i, img)
                        self.draw.updBufferf('clr%d' % i, img.reshape(-1, 3))
                    self.draw.updBufferf('pts%d' % i, points[(t, i)][0])
                    if points_normals is not None:
                        self.draw.updBufferf('pts_nrm%d' % i, points_normals[(t, i)][0])
                        img_nrm = viz_normals(data['normals'][(t, i)][0]).reshape(-1, 3)
                        img_nrm2 = np.zeros((img_nrm.shape[0] * 2, img_nrm.shape[1]))
                        img_nrm2[::2], img_nrm2[1::2] = img_nrm, img_nrm
                        self.draw.updBufferf('clr_nrm%d' % i, img_nrm2)

            self.draw.clear()
            for i, (cam_key, cam_val) in enumerate(camcv.items()):
                if cam_key[0] == t:
                    self.draw['wld'].size(2).color(
                        self.cam_colors[cam_key[1]]).points(
                        'pts%d' % cam_key[1], ('clr%d' % cam_key[1]) if color else None)
                    if show_normals:
                        self.draw['wld'].width(1).lines(
                            'pts_nrm%d' % cam_key[1], 'clr_nrm%d' % cam_key[1])
                self.draw['img%d%d' % modrem(cam_key[1], 2)].image('img%d' % cam_key[1])
                clr = self.cam_colors[cam_key[1]] if cam_key[0] == t else 'gra'
                tex = 'img%d' % cam_key[1] if cam_key[0] == t else None
                self.draw['wld'].object(cam_val, color=clr, tex=tex)
            text = keys[k].upper() + ('' if offsets[k] is None else '_FWD' if offsets[k] == 1 else '_BWD')
            self.draw['wld'].text(f'{text}   ({t})', (0,0))

            if actions is not None:
                for i, cam_key in enumerate(camcv.keys()):
                    if cam_key[0] == t:
                        for j in range(actions[t].shape[1]):
                            string = str('%4.4f' % actions[t][0, j].numpy())
                            w = 900
                            if string.startswith('-'): w -= 14
                            self.draw['wld'].text(string, (w, j + 1))
            if language is not None:
                for i, cam_key in enumerate(camcv.keys()):
                    if cam_key[0] == t:
                        for j in range(len(language[t])):
                            string = language[t][j]
                            self.draw['wld'].text(string, (0, j + 2))

            self.draw.update(30)
