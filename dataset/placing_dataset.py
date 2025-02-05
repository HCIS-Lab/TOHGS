from torch.utils.data import Dataset
import torch
import os
import pickle
from torchvision import transforms
import numpy as np
# from utils import utils
import time
from PIL import Image
import json
import pytorch3d.ops
from torch.utils.data import DataLoader
import glob
from scipy.spatial.transform import Rotation as Rot
from manopth.manolayer import ManoLayer
from manopth.rodrigues_layer import batch_rodrigues
import trimesh


class grasping_pose(Dataset):
    def __init__(self, mode = "train", batch_size=160, root = 'task_oriented_grasps_dataset/placing', sample_points = 3000):
        self.root = root
        self.batch_size = batch_size
        self.mode = mode
        self.obj_list = []
        self.sample_points = sample_points
        if self.mode == 'train':
            with open(os.path.join(root, 'train.txt'), 'r') as file:
                Lines = file.readlines()
                for line in Lines:
                    self.obj_list.append(line.strip())
        elif self.mode == 'val':
            with open(os.path.join(root, 'val.txt'), 'r') as file:
                Lines = file.readlines()
                for line in Lines:
                    self.obj_list.append(line.strip())
        self.init_pose = []
        self.goal_pose = []
        self.recon_param = []
        self.obj_name = []
        for obj in self.obj_list:
            path = os.path.join(self.root, obj)
            config = np.load(path, allow_pickle = True).item()
            for i in range(len(config['init_pose'])):
                self.init_pose.append(config['init_pose'][i])
                self.goal_pose.append(config['goal_pose'][i])
                self.recon_param.append(config['recon_param'][i])
                self.obj_name.append(obj.split('.')[0])
        self.dataset_size = len(self.init_pose)
        self.manolayer = ManoLayer(mano_root='./models/mano/',
                              flat_hand_mean=True, use_pca=False)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        init_grasped_obj_pose = self.init_pose[idx]
        recon_param = self.recon_param[idx]
        grasped_obj_name = self.obj_name[idx]
        grasped_obj_mesh = trimesh.load_mesh(os.path.join('models', 'meshdata', grasped_obj_name, 'coacd', 'decomposed_scaled.obj'))
        init_grasped_pose = np.eye(4)
        init_grasped_pose[:3, :3] = Rot.from_quat(
            init_grasped_obj_pose[3:]).as_matrix()
        init_grasped_pose[:3, 3] = init_grasped_obj_pose[:3].T
        sample = trimesh.sample.sample_surface(grasped_obj_mesh, self.sample_points)[0]
        sample = np.matmul(init_grasped_pose[:3, :3], sample.T) + init_grasped_pose[:3, 3].reshape(-1, 1)
        trans = np.mean(sample, axis=1)
        sample -= np.tile(trans, (self.sample_points,1)).T
        recon_param[:,:3] -= trans
        hand_pose_init = torch.FloatTensor(recon_param)
        recon_param = torch.FloatTensor(recon_param)
        hand_vertex, _ = self.manolayer.forward(
                th_trans=recon_param[:, :3],
                th_pose_coeffs=recon_param[:, 3:],
            )
        if self.mode == 'train':
            hand_mat = np.eye(4)
            orient = torch.FloatTensor(1, 3).uniform_(-np.pi, np.pi)
            aug_rot_mats = batch_rodrigues(orient.view(-1, 3)).view([1, 3, 3])
            aug_rot_mat = aug_rot_mats[0]
            aug_trans = torch.eye(4)
            aug_trans[:3, :3] = aug_rot_mat
            recon_param_original = recon_param.clone()
            glob_orient = Rot.from_rotvec(hand_pose_init[:, 3:6]).as_matrix()
            hand_mat[:3, :3] = glob_orient
            hand_mat[:3, 3] = hand_pose_init[:, :3].squeeze(0).numpy() + self.manolayer.wrist_trans.numpy()
            hand_mat = aug_trans @ hand_mat
            recon_param_original[:, :3] = hand_mat[:3, 3].clone().float() - self.manolayer.wrist_trans.clone()
            recon_param_original[:, 3:6] = torch.tensor(Rot.from_matrix(hand_mat[:3, :3]).as_rotvec()).float()
            hand_vertex, _ = self.manolayer.forward(
                    th_trans=recon_param_original[:, :3],
                    th_pose_coeffs=recon_param_original[:, 3:],
                )
            sample = np.matmul(aug_rot_mat.numpy(), sample)
            # furthest_distance = np.max(np.sqrt(np.sum(abs(sample)**2,axis=0)))
            # normalized_sample = sample / furthest_distance
            return {
                "hand_verts": hand_vertex.squeeze(0) / 1000,
                "obj_verts": torch.FloatTensor(sample.T),
                "recon_param": recon_param_original.squeeze(0),
            }
        elif self.mode == 'val':
            return {
                "hand_verts": hand_vertex.squeeze(0) / 1000,
                "obj_verts": torch.FloatTensor(sample.T),
                "recon_param": recon_param.squeeze(0),
            }


class scene(Dataset):
    def __init__(self, mode = "train", batch_size=160, root = 'task_oriented_grasps_dataset/placing', sample_points = 3000):
        self.root = root
        self.batch_size = batch_size
        self.mode = mode
        self.obj_list = []
        self.sample_points = sample_points
        if self.mode == 'train':
            with open(os.path.join(root, 'train.txt'), 'r') as file:
                Lines = file.readlines()
                for line in Lines:
                    self.obj_list.append(line.strip())
        elif self.mode == 'val':
            with open(os.path.join(root, 'val.txt'), 'r') as file:
                Lines = file.readlines()
                for line in Lines:
                    self.obj_list.append(line.strip())
        self.init_pose = []
        self.goal_pose = []
        self.recon_param = []
        self.obj_name = []
        self.obstacle_pose = []
        self.obstacle_name = []
        for obj in self.obj_list:
            path = os.path.join(self.root, obj)
            config = np.load(path, allow_pickle = True).item()
            for i in range(len(config['init_pose'])):
                self.init_pose.append(config['init_pose'][i])
                self.goal_pose.append(config['goal_pose'][i])
                self.recon_param.append(config['recon_param'][i])
                self.obstacle_pose.append(config['obstacle_pose'][i])
                self.obstacle_name.append(config['obstacle_name'][i])
                self.obj_name.append(obj.split('.')[0])
        self.dataset_size = len(self.init_pose)
        self.manolayer = ManoLayer(mano_root='./models/mano/',
                              flat_hand_mean=True, use_pca=False)
        self.table = trimesh.load_mesh(os.path.join('models', 'table', 'table.obj'))
        self.table_height = 0.6
        table_pose = np.eye(4)
        table_pose[:3, :3] = Rot.from_quat(
            np.array([0, 0, 0, 1])).as_matrix()
        table_pose[:3, 3] = np.array([0, 0, self.table_height]).T
        self.table.apply_scale([1.5, 1, 0.05])
        self.table.apply_transform(table_pose)    

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        init_grasped_obj_pose = self.init_pose[idx]
        goal_grasped_obj_pose = self.goal_pose[idx]
        recon_param = self.recon_param[idx]
        grasped_obj_name = self.obj_name[idx]
        obstacle_pose = self.obstacle_pose[idx]
        obstacle_name = self.obstacle_name[idx]
        grasped_obj_mesh = trimesh.load_mesh(os.path.join('models', 'meshdata', grasped_obj_name, 'coacd', 'decomposed_scaled.obj'))

        init_grasped_pose = np.eye(4)
        init_grasped_pose[:3, :3] = Rot.from_quat(
            init_grasped_obj_pose[3:]).as_matrix()
        init_grasped_pose[:3, 3] = init_grasped_obj_pose[:3].T

        goal_grasped_pose = np.eye(4)
        goal_grasped_pose[:3, :3] = Rot.from_quat(
            goal_grasped_obj_pose[3:]).as_matrix()
        goal_grasped_pose[:3, 3] = goal_grasped_obj_pose[:3].T

        grasped_obj_pcd = trimesh.sample.sample_surface(grasped_obj_mesh, self.sample_points)[0]
        init_grasped_obj_pcd = np.matmul(init_grasped_pose[:3, :3], grasped_obj_pcd.T) + init_grasped_pose[:3, 3].reshape(-1, 1)
        init_trans = np.mean(init_grasped_obj_pcd.T, axis=0).reshape(1, 3)
        origin_grasped_obj_pcd = torch.FloatTensor(init_grasped_obj_pcd.T - init_trans).T
        goal_grasped_obj_pcd = np.matmul(goal_grasped_pose[:3, :3], grasped_obj_pcd.T) + goal_grasped_pose[:3, 3].reshape(-1, 1)
        table_pcd = trimesh.sample.sample_surface(self.table, self.sample_points)[0]
        # goal_trans = np.mean(goal_grasped_obj_pcd, axis=1)
        recon_param = torch.FloatTensor(recon_param)
        init_hand_vertex, _ = self.manolayer.forward(
                th_trans=recon_param[:, :3],
                th_pose_coeffs=recon_param[:, 3:],
            )
        init_hand_vertex = init_hand_vertex[0].T
        recon_param_original = recon_param.clone()
        # recon_param_original[:, :3] -= torch.FloatTensor(init_trans).reshape(-1)
        obstacles_pcd_list = [torch.FloatTensor(table_pcd)]
        for i in range(len(obstacle_name)):
            name = obstacle_name[i]
            temp = np.array(obstacle_pose[i])
            pose = np.eye(4)
            pose[:3, :3] = Rot.from_quat(
                temp[3:]).as_matrix()
            pose[:3, 3] = temp[:3].T
            mesh = trimesh.load(os.path.join('models', 'meshdata', name, 'coacd', 'decomposed_scaled.obj'))
            mesh.apply_transform(pose)
            sample = trimesh.sample.sample_surface(mesh, self.sample_points)[0]
            obstacles_pcd_list.append(torch.FloatTensor(sample))
       
        scene_pc = torch.cat(obstacles_pcd_list, dim=0)
        scene_pc_fps = pytorch3d.ops.sample_farthest_points(scene_pc.unsqueeze(0), K=6000)[0][0].T  # [NO, 3]
        
        if self.mode == 'train':
            orient = torch.FloatTensor(1, 3).uniform_(-np.pi, np.pi)
            hand_mat = np.eye(4)
            aug_rot_mats = batch_rodrigues(orient.view(-1, 3)).view([1, 3, 3])
            aug_rot_mat = aug_rot_mats[0]
            aug_trans = torch.eye(3)
            aug_trans_homo = torch.eye(4)
            aug_trans[:3, :3] = aug_rot_mat
            aug_trans_homo[:3, :3] = aug_rot_mat 
            init_hand_vertex = aug_trans @ init_hand_vertex
            init_grasped_obj_pcd = aug_trans.numpy() @ init_grasped_obj_pcd
            origin_grasped_obj_pcd = aug_trans @ origin_grasped_obj_pcd
            goal_grasped_obj_pcd = aug_trans.numpy() @ goal_grasped_obj_pcd
            scene_pc_fps = aug_trans @ scene_pc_fps
            glob_orient = Rot.from_rotvec(recon_param_original[:, 3:6]).as_matrix()
            hand_mat[:3, :3] = glob_orient
            hand_mat[:3, 3] = recon_param_original[:, :3].squeeze(0).numpy() + self.manolayer.wrist_trans.numpy()
            hand_mat = aug_trans_homo @ hand_mat
            recon_param_original[:, :3] = hand_mat[:3, 3].clone().float() - self.manolayer.wrist_trans.clone()
            recon_param_original[:, 3:6] = torch.tensor(Rot.from_matrix(hand_mat[:3, :3]).as_rotvec()).float()
            original_hand_vertex, _ = self.manolayer.forward(
                th_trans=recon_param_original[:, :3],
                th_pose_coeffs=recon_param_original[:, 3:],
            )
            origin_hand_vertex = original_hand_vertex[0].T
            
            
        return {
                "init_hand_verts": init_hand_vertex.T / 1000,
                # "origin_hand_verts": origin_hand_vertex.T / 1000,
                "init_obj_verts": torch.FloatTensor(init_grasped_obj_pcd.T),
                "goal_obj_verts": torch.FloatTensor(goal_grasped_obj_pcd.T),
                "origin_obj_verts": torch.FloatTensor(origin_grasped_obj_pcd.T),
                "scene_pc": scene_pc_fps.T,
                "recon_param": recon_param_original.squeeze(0)
            }
        
if __name__ == '__main__':
    dataset = scene(sample_points=3000)
    dataloder = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)
    for i, (data) in enumerate(dataloder):
        color1 = np.full((3000, 4), 255)
        color1[:, 0] = 0
        color2 = np.full((3000, 4), 255)
        color2[:, 1] = 0
        color3 = np.full((3000, 4), 255)
        color3[:, 2] = 0
        color4 = np.full((3000, 4), 255)
        pct_1 = trimesh.PointCloud(data['init_hand_verts'][0].numpy(), colors=color1)
        pct_2 = trimesh.PointCloud(data['init_obj_verts'][0].numpy(), colors=color2)
        pct_3 = trimesh.PointCloud(data['scene_pc'][0].numpy(), colors=color3)
        pct_4 = trimesh.PointCloud(data['origin_hand_verts'][0].numpy(), colors=color2)
        pct_5 = trimesh.PointCloud(data['origin_obj_verts'][0].numpy(), colors=color3)
        pct_6 = trimesh.PointCloud(data['goal_obj_verts'][0].numpy(), colors=color4)
        print(i)
        trimesh.Scene([pct_1, pct_2, pct_3, pct_6]).show()
        trimesh.Scene([pct_4, pct_5]).show()