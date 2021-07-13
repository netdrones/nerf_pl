import os
import torch
import imageio
import numpy as np
from collections import defaultdict

from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

from models.nerf import *
from models.rendering import *

from utils import load_ckpt, visualize_depth

from datasets.ray_utils import *
from datasets.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary, qvec2rotmat

# HYPERPARAMETERS
#########################
N_vocab = 1500
N_a = 48
N_tau = 16
beta_min = 0.03
N_emb_xyz = 10
N_emb_dir = 4
NEAR = 0
FAR = 5
#########################

def coord_from_pose(P):
    bottom = np.array([0,0,0,1]).reshape(1,4)
    P = np.concatenate([P, bottom], 0)
    M = np.linalg.inv(P)[:3]
    R = M[:, :3]
    t = M[:, :4]
    c = np.linalg.inv(R) @ t

    return c

def circ_rot(pose, radius, phi, theta):
    coord = coord_from_pose(pose)

    translation = np.array([
        [1,0,0,0],
        [0,1,0,-0.9*radius],
        [0,0,1,radius],
        [0,0,0,1],
    ])

    rot_phi = np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi),np.cos(phi),0],
        [0,0,0,1],
    ])

    rot_theta = np.array([
        [np.cos(theta),0,-np.sin(theta),0],
        [0,1,0,0],
        [np.sin(theta),0,np.cos(theta),0],
        [0,0,0,1],
    ])

    T = rot_theta @ rot_phi @ translation
    result = coord @ T
    return result[:3]

def generate_spheric_poses(pose, radius, n_poses):
    poses = []
    for th in np.linspace(0,2*np.pi, n_poses+1)[:-1]:
        pose = circ_rot(pose, radius, -np.pi/5, th)
        poses += [pose]

    return np.stack(poses, 0)

def render_circle(render, idx, radius, n_frames):
    sample = render[idx]
    pose = sample['c2w']

    # Define testing intrinsics
    render.test_appearance_idx = idx
    render.test_img_w, render.test_img_h = sample['img_wh']
    render.test_focal = render.test_img_w/2/np.tan(np.pi/6)  # 60 FOV
    render.test_K = np.array([[render.test_focal, 0, render.test_img_w/2],
                             [0, render.test_focal, render.test_img_h/2],
                             [0, 0, 1]])
    render.poses_test = generate_spheric_poses(pose, radius, n_frames)
    res_list = []

    for i in tqdm(range(len(render))):
        sample = render[i]
        rays = sample['rays']
        ts = sample['ts']
        results = f(rays.cuda(), ts.cuda(), render)
        res_list.append(results)

    return res_list

def png_from_idx(render, idx):
    w, h = render[idx]['img_wh']
    results = predict_image_from_idx(render, idx)
    img_pred = np.clip(results['rgb_fine'].view(h,w,3).cpu().numpy(), 0, 1)
    img_pred_ = (img_pred*255).astype(np.uint8)
    imageio.imwrite(f'{idx:03d}.png', img_pred_)

@torch.no_grad()
def f(rays, ts, render, N_samples=64, N_importance=64, use_disp=False, chunk=1024*32, white_back=False, **kwargs):
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = render_rays(render.models,
                                          render.embeddings,
                                          rays[i:i+chunk],
                                          ts[i:i+chunk],
                                          N_samples,
                                          False,
                                          0,
                                          0,
                                          N_importance,
                                          chunk,
                                          False,
                                          test_time=True,
                                          **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]
    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    return results

def predict_image_from_idx(render, idx):

    sample = render[idx]
    rays = sample['rays'].cuda()
    ts = sample['ts'].cuda()
    results = f(rays, ts, render)

    return results

class Render:
    def __init__(self, ckpt_path, colmap_path=None, split='test', img_downscale=1):
        """
        ckpt_path: Path to model checkpoints
        colmap_path: Path to computed COLMAP dense parameters
        img_downscale: Degree to which rendered images are downsampled (adjust if running into OOM errors)
        """
        self.split = split
        self.poses_test = {}
        self.ckpt_path = ckpt_path
        self.transform = T.ToTensor()
        self.colmap_path = colmap_path
        self.img_downscale = img_downscale

        self.construct_model()

        if self.colmap_path:
            self.load_poses()
            self.load_cameras()

    def construct_model(self):
        embedding_xyz = PosEmbedding(N_emb_xyz-1, N_emb_xyz)
        embedding_dir = PosEmbedding(N_emb_dir-1, N_emb_dir)
        embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
        embedding_a = torch.nn.Embedding(N_vocab, N_a).cuda()
        load_ckpt(embedding_a, self.ckpt_path, model_name='embedding_a')
        embeddings['a'] = embedding_a
        embedding_t = torch.nn.Embedding(N_vocab, N_tau).cuda()
        load_ckpt(embedding_t, self.ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t
        nerf_coarse = NeRF('coarse',
                           in_channels_xyz=6*N_emb_xyz+3,
                           in_channels_dir=6*N_emb_dir+3).cuda()
        nerf_fine = NeRF('fine',
                         in_channels_xyz=6*N_emb_xyz+3,
                         in_channels_dir=6*N_emb_dir+3,
                         encode_appearance=True,
                         in_channels_a=N_a,
                         encode_transient=True,
                         in_channels_t=N_tau,
                         beta_min=beta_min).cuda()

        load_ckpt(nerf_coarse, self.ckpt_path, model_name='nerf_coarse')
        load_ckpt(nerf_fine, self.ckpt_path, model_name='nerf_fine')

        self.embeddings = embeddings
        self.models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    def load_poses(self):
        self.imdata = read_images_binary(os.path.join(self.colmap_path, 'dense/sparse/images.bin'))

        # Store two-way mapping of image paths to COLMAP image ID
        self.img_path_to_id = {}
        for v in self.imdata.values():
            self.img_path_to_id[v.name] = v.id
        self.img_ids = []
        self.image_paths = {}
        for filename in self.img_path_to_id.keys():
            id_ = self.img_path_to_id[filename]
            self.image_paths[id_] = filename
            self.img_ids += [id_]

        # Load poses
        w2c_mats = []
        bottom = np.array([0,0,0,1]).reshape(1,4)
        for id_ in self.img_ids:
            img = self.imdata[id_]
            R = img.qvec2rotmat()
            t = img.tvec.reshape(3,1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        self.poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4)
        self.poses[..., 1:3] *= -1

        # Normalize scale
        pts_3d = read_points3d_binary(os.path.join(self.colmap_path, 'dense/sparse/points3D.bin'))
        self.xyz_world = np.array([pts_3d[p_id].xyz for p_id in pts_3d])
        xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
        self.nears, self.fars = {}, {}  # {id: distance}
        for i, id_ in enumerate(self.img_ids):
            xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3]  # XYZ at i'th camera coordinate
            xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] > 0]  # Filter out points lying behind the camera
            self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
            self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

        max_far = np.fromiter(self.fars.values(), np.float32).max()
        scale_factor = max_far / 5
        self.poses[..., 3] /= scale_factor
        for k in self.nears:
            self.nears[k] /= scale_factor
        for k in self.fars:
            self.fars[k] /= scale_factor

        self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

    def load_cameras(self):
        self.Ks = {}
        self.camdata = read_cameras_binary(os.path.join(self.colmap_path, 'dense/sparse/cameras.bin'))

        # Construct camera intrinsic matrices (K)
        for key in self.camdata:
            K = np.zeros((3,3), dtype=np.float32)
            cam = self.camdata[key]
            W, H = int(cam.params[2]*2), int(cam.params[3]*2)
            W_scaled, H_scaled = W // self.img_downscale, H // self.img_downscale

            K[0,0] = cam.params[0] * (W_scaled/W)  # f_x
            K[1,1] = cam.params[1] * (H_scaled/H)  # f_y
            K[0,2] = cam.params[2] * (W_scaled/W)  # c_x
            K[1,2] = cam.params[3] * (H_scaled/H)  # c_y
            K[2,2] = 1

            self.Ks[key] = K

    def __len__(self):
        if len(self.poses_test) > 0:
            return len(self.poses_test)
        else:
            return len(self.poses_dict)

    def __getitem__(self, idx):
        sample = {}

        if self.split == 'val':
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[idx])
            img = Image.open(os.path.join(self.colmap_path, 'dense/images',
                                          self.image_paths[idx])).convert('RGB')

            # Downscale the image
            img_w, img_h = img.size
            img_w = img_w // self.img_downscale
            img_h = img_h // self.img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)

            # Store RGB values
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
            sample['rgbs'] = img

            # Compute rays
            directions = get_ray_directions(img_h, img_w, self.Ks[idx])
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = NEAR, FAR
            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1)  # (h*w, 8)
            sample['rays'] = rays
            sample['ts'] = idx * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])

        elif self.split == 'test':
            if len(self.poses_test) > 0 :
                sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
                directions = get_ray_directions(self.test_img_h, self.test_img_w, self.test_K)
                rays_o, rays_d = get_rays(directions, c2w)
                near, far = NEAR, FAR
                rays = torch.cat([rays_o, rays_d,
                                  near*torch.ones_like(rays_o[:, :1]),
                                  far*torch.ones_like(rays_o[:, :1])],
                                  1)
                sample['rays'] = rays
                sample['ts'] = self.test_appearance_idx * torch.ones(len(rays), dtype=torch.long)
                sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])

            else:
                sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[idx])
                img = Image.open(os.path.join(self.colmap_path, 'dense/images',
                                              self.image_paths[idx])).convert('RGB')

                # Downscale the image
                img_w, img_h = img.size
                img_w = img_w // self.img_downscale
                img_h = img_h // self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)

                # Store RGB values
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
                sample['rgbs'] = img

                # Compute rays
                directions = get_ray_directions(img_h, img_w, self.Ks[self.imdata[idx].camera_id])
                rays_o, rays_d = get_rays(directions, c2w)
                near, far = NEAR, FAR
                rays = torch.cat([rays_o, rays_d,
                                  near*torch.ones_like(rays_o[:, :1]),
                                  far*torch.ones_like(rays_o[:, :1])],
                                  1)  # (h*w, 8)
                sample['rays'] = rays
                sample['ts'] = idx * torch.ones(len(rays), dtype=torch.long)
                sample['img_wh'] = torch.LongTensor([img_w, img_h])

        return sample
