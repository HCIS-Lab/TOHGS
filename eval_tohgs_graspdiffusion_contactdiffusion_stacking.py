import torch
import os
import torch
import argparse
import numpy as np
import trimesh
from metric.simulate import run_simulation
from scipy.spatial.transform import Rotation as Rot
import datetime
from manopth.manolayer import ManoLayer
from network.grasp_unet import UNetModel
from scheduler.ddpm import DDPM as GraspDiffusion
from scheduler.ddpm_task_contact import DDPM as ContactDiffusion
from itertools import combinations
import pytorch3d.ops
import json
import kaolin.ops
import time
import cv2
def euclidean_dist(hand, obj, normalized = True, alpha = 100):
    batch_object_point_cloud = obj.unsqueeze(1)
    batch_object_point_cloud = batch_object_point_cloud.repeat(1, hand.size(1), 1, 1).transpose(1, 2)
    hand_surface_points = hand.unsqueeze(1)
    hand_surface_points = hand_surface_points.repeat(1, obj.size(1), 1, 1)
    object_hand_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
    contact_dist = object_hand_dist.min(dim=2)[0]
    if normalized:
        contact_value_current = 1 - 2 * (torch.sigmoid(alpha * contact_dist) - 0.5)
        return contact_value_current
    else:
        return contact_dist


def intersect_vox(obj_mesh, hand_mesh, pitch=0.001):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    try:
        obj_vox = obj_mesh.voxelized(pitch=pitch)
        obj_points = obj_vox.points
        inside = hand_mesh.contains(obj_points)
        volume = inside.sum() * np.power(pitch, 3)
    except:
        volume = 1
    return volume


def modified_intersect_vox(obj_vox, hand_mesh, pitch=0.001):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    try:
        obj_points = obj_vox.points
        inside = hand_mesh.contains(obj_points)
        volume = inside.sum() * np.power(pitch, 3)
    except:
        volume = 1
    return volume


def gpu_intersect_vox(verts, faces,  point, pitch=0.001):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    inside = kaolin.ops.mesh.check_sign(verts, faces, point)
    volume = inside.sum(dim = 1)
    return volume * np.power(pitch, 3)


def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign


def fit_sigmoid(colors, a=0.05):
    '''Fits a sigmoid to raw contact temperature readings from the ContactPose dataset. This function is copied from that repo'''
    idx = colors > 0
    ci = colors[idx]

    x1 = torch.min(ci)  # Find two points
    y1 = a
    x2 = torch.max(ci)
    y2 = 1-a

    lna = np.log((1 - y1) / y1)
    lnb = np.log((1 - y2) / y2)
    k = (lnb - lna) / (x1 - x2)
    mu = (x2*lna - x1*lnb) / (lna - lnb)
    ci = torch.exp(k * (ci-mu)) / (1 + torch.exp(k * (ci-mu)))  # Apply the sigmoid
    colors[idx] = ci
    return colors


def overall_diversity(total_grasp):
    if len(total_grasp) != 1:
        comb = combinations([i for i in range(len(total_grasp))], 2)
        comb = list(comb)
        total_dis = 0
        for i in comb:
            norm = np.linalg.norm(total_grasp[i[0]] - total_grasp[i[1]], axis = 1)
            total_dis += norm.sum()
        return total_dis / len(comb)
    else:
        return 0


def get_json(filename):
    f = open(filename)
    data = json.load(f)
    goal = []
    for i in data['goal']:
        goal.append(i)
    init = []
    for i in data['init']:
        init.append(i)
    # Closing file
    f.close()
    return init, goal


def umnormalize_to_zero_and_one(img):
    return (img + 1) / 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type = int, default = 1)
    parser.add_argument('--grasp_model_path', type=str, default='')
    parser.add_argument("--grasp_model_diffusion_step", type=int, default=1000)
    parser.add_argument("--grasp_model_scheduler", type=str, default='linear')
    parser.add_argument('--contact_model_path', type=str, default='')
    parser.add_argument("--contact_model_diffusion_step", type=int, default=1000)
    parser.add_argument("--contact_model_scheduler", type=str, default='linear')
    parser.add_argument('--mode', type=str, default='mu')
    parser.add_argument('--cmap_sample_num', type=int, default=16)
    parser.add_argument('--grasp_sample_num', type=int, default=1)
    parser.add_argument('--task', type=str, default='stacking')
    parser.add_argument('--test_data_path', type=str, default='task_oriented_grasps_dataset')
    parser.add_argument('--point_sample_num', type=int, default=2048)
    parser.add_argument('--set', type=str, default='eval')
    parser.add_argument('--penetr_vol_thre', type = float, default = 6e-6)  # 6cm^3
    parser.add_argument('--simu_disp_thre', type = float, default = 0.04)  # 4cm
    parser.add_argument('--collision_thre', type = float, default = 1e-7)  # 4cm^3
    parser.add_argument('--vis', default=False, action='store_true')
    parser.add_argument('--opt', default=False, action='store_true')
    parser.add_argument('--alpha', type=int, default=30)
    args = parser.parse_args()
    assert args.task in ['placing', 'stacking']
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    contact_model = ContactDiffusion(args.contact_model_diffusion_step, args.contact_model_scheduler, args.alpha).float()
    checkpoints = torch.load(args.contact_model_path)["model_state"]
    new_state_dict = {}
    for key in checkpoints:
        new_key = key.replace('module.','')
        new_state_dict[new_key] = checkpoints[key]
    contact_model.load_state_dict(new_state_dict)
    contact_model = contact_model.to(device).eval()

    grasp_model = GraspDiffusion(UNetModel(), timesteps=args.grasp_model_diffusion_step, beta_schedule=args.grasp_model_scheduler, optimize=args.opt)
    checkpoints = torch.load(args.grasp_model_path)["model_state"]
    grasp_model.load_state_dict(checkpoints)
    grasp_model = grasp_model.to(device).eval()

    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d_%H:%M:%S")
    with torch.no_grad():
        rh_mano = ManoLayer(mano_root='./models/mano/',
                              flat_hand_mean=True, use_pca=False).to(device=device)
    rh_faces = rh_mano.th_faces.view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes
    config_path = os.path.join(args.test_data_path, args.task)
    obj_list = []
    if args.set == 'train':
        with open(os.path.join(config_path, 'train.txt'), 'r') as file:
            Lines = file.readlines()
            for line in Lines:
                obj_list.append(line.strip())
    elif args.set =='eval':
        with open(os.path.join(config_path, 'val.txt'), 'r') as file:
            Lines = file.readlines()
            for line in Lines:
                obj_list.append(line.strip())
    table = trimesh.load('models/table/table.obj')
    table_pose = np.eye(4)
    table_pose[:3, :3] = Rot.from_quat(
        np.array([0, 0, 0, 1])).as_matrix()
    table_pose[:3, 3] = np.array([0, 0, 0.6]).T
    table.apply_scale([1.5, 1, 0.05])
    table.apply_transform(table_pose)
    h2o_consistency_score = []
    penetration_volume = []
    simulation_displacement = []
    contact_ratio = []
    diversity_score = []
    init_scene_penetration_percentage = []
    goal_scene_penetratoin_percentage = []
    good_ratio = []
    succenssful_ratio = []
    total_count = 0
    local_time = time.localtime(time.time())
    time_str = str(local_time[1]) + '_' + str(local_time[2]) + '_' + str(local_time[3]) + '_' + str(local_time[4]) + '_' + str(local_time[5])
    dirname = 'result/tohgs_graspdiffusion_contactdiffusion_stacking_{}'.format(time_str)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        os.makedirs(os.path.join(dirname, 'image'))
    file = open(os.path.join(dirname ,'result.txt'), 'w')
    print(list(vars(args).items()), file=file)
    file.flush()
    qr_threshold = [[4e-6, 0.03],[4e-6, 0.02],[4e-6, 0.01],
                    [3e-6, 0.03],[3e-6, 0.02],[3e-6, 0.01],
                    [2e-6, 0.03],[2e-6, 0.02],[2e-6, 0.01],
                    [1e-6, 0.03],[1e-6, 0.02],[1e-6, 0.01]]
    qr = [[],[],[],[],[],[],[],[],[],[],[],[]]
    print(qr_threshold)
    for k, obj in enumerate(obj_list):
        config = os.path.join(config_path, 'test', obj.replace('.npy', '.json'))
        init, goal = get_json(config)
        obj_h2o_consistency_score = []
        obj_penetration_volume = []
        obj_simulation_displacement = []
        obj_contact_ratio = []
        obj_diversity_score = []
        obj_init_scene_penetration_percentage = []
        obj_goal_scene_penetratoin_percentage = []
        obj_succenssful_ratio = []
        obj_good_ratio = []
        print('obj: {}'.format(obj))
        output = {'init_hand_vertex':[], 'goal_hand_vertex':[], 'init_pose':[], 'goal_pose':[]}
        for i in range(len(init)):
            init_config = init[i]
            goal_config = goal[i]
            based_obj_name = goal_config[0]
            grasped_obj_name = goal_config[2]
            goal_based_obj_pose = np.array(goal_config[1])
            goal_grasped_obj_pose = np.array(goal_config[3])
            init_based_obj_pose = np.array(init_config[1])
            init_grasped_obj_pose = np.array(init_config[3])
            grasped_obj_mesh = trimesh.load_mesh(os.path.join('models', 'brick', grasped_obj_name, 'textured_simple.obj'))
            based_obj_mesh = trimesh.load_mesh(os.path.join('models', 'brick', based_obj_name, 'textured_simple.obj'))

            init_grasped_pose = np.eye(4)
            init_grasped_pose[:3, :3] = Rot.from_quat(
                init_grasped_obj_pose[3:]).as_matrix()
            init_grasped_pose[:3, 3] = init_grasped_obj_pose[:3].T

            goal_grasped_pose = np.eye(4)
            goal_grasped_pose[:3, :3] = Rot.from_quat(
                goal_grasped_obj_pose[3:]).as_matrix()
            goal_grasped_pose[:3, 3] = goal_grasped_obj_pose[:3].T
            
            init_based_pose = np.eye(4)
            init_based_pose[:3, :3] = Rot.from_quat(
                init_based_obj_pose[3:]).as_matrix()
            init_based_pose[:3, 3] = init_based_obj_pose[:3].T

            goal_based_pose = np.eye(4)
            goal_based_pose[:3, :3] = Rot.from_quat(
                goal_based_obj_pose[3:]).as_matrix()
            goal_based_pose[:3, 3] = goal_based_obj_pose[:3].T

            init_grasped_obj_mesh = trimesh.load(os.path.join('models', 'brick', grasped_obj_name, 'textured_simple.obj'))
            init_grasped_obj_mesh.apply_transform(init_grasped_pose)
            init_grasped_obj_vox = init_grasped_obj_mesh.voxelized(pitch=0.001)
            goal_grasped_obj_mesh = trimesh.load(os.path.join('models', 'brick', grasped_obj_name, 'textured_simple.obj'))
            goal_grasped_obj_mesh.apply_transform(goal_grasped_pose)
            
            init_grasped_obj_mesh_pcd = trimesh.sample.sample_surface(init_grasped_obj_mesh, args.point_sample_num)[0]
            temp = np.matmul(np.linalg.inv(init_grasped_pose[:3, :3]), (init_grasped_obj_mesh_pcd.T - init_grasped_pose[:3, 3].reshape(3, 1)))
            goal_grasped_obj_mesh_pcd = (np.matmul(goal_grasped_pose[:3, :3], temp) + goal_grasped_pose[:3, 3].reshape(3, 1)).T
            
            
            init_based_obj_mesh = trimesh.load(os.path.join('models', 'brick', based_obj_name, 'textured_simple.obj'))
            init_based_obj_mesh.apply_transform(init_based_pose)
            # init_based_obj_vox = init_based_obj_mesh.voxelized(pitch=0.001)
            # goal_based_obj_mesh = trimesh.load(os.path.join('models', 'meshdata', grasped_obj_name, 'coacd', 'decomposed_scaled.obj'))
            # goal_based_obj_mesh.apply_transform(goal_based_pose)
            

            table_pcd = trimesh.sample.sample_surface(table, args.point_sample_num)[0]
            obstacles_pcd_list = [torch.FloatTensor(table_pcd)]
            obstacle_mesh = [pytorch3d.structures.Meshes(torch.FloatTensor(table.vertices).unsqueeze(0), torch.FloatTensor(table.faces).unsqueeze(0)).cuda()]
            obs_mesh = [table]
            
            sample = trimesh.sample.sample_surface(init_based_obj_mesh, args.point_sample_num)[0]
            obstacles_pcd_list.append(torch.FloatTensor(sample))
            pyt3d_mesh = pytorch3d.structures.Meshes(torch.FloatTensor(init_based_obj_mesh.vertices).unsqueeze(0), torch.FloatTensor(init_based_obj_mesh.faces).unsqueeze(0)).cuda()
            obstacle_mesh.append(pyt3d_mesh)
            obs_mesh.append(init_based_obj_mesh)

            scene_mesh = pytorch3d.structures.join_meshes_as_scene(obstacle_mesh)
            obstacle_pcd = torch.cat(obstacles_pcd_list, dim=0).unsqueeze(0).to(device)
            scene_pcd = pytorch3d.ops.sample_farthest_points(obstacle_pcd, K=6000)[0]
            obstacle_pcd = torch.cat(obstacles_pcd_list, dim=0).unsqueeze(0).to(device)
            init_grasped_obj_mesh_pcd = torch.FloatTensor(init_grasped_obj_mesh_pcd).unsqueeze(0).to(device)
            goal_grasped_obj_mesh_pcd = torch.FloatTensor(goal_grasped_obj_mesh_pcd).unsqueeze(0).to(device)
            trans = torch.mean(init_grasped_obj_mesh_pcd[0], dim = 0).reshape(-1, 3)
            origin_grasped_obj_mesh_pcd = (init_grasped_obj_mesh_pcd[0] - trans.repeat(args.point_sample_num, 1)).unsqueeze(0)
            origin_obj_normal = pytorch3d.ops.estimate_pointcloud_normals(origin_grasped_obj_mesh_pcd)
            obj_pc = torch.cat([origin_grasped_obj_mesh_pcd, origin_obj_normal], dim = 2)
            obj_pc = obj_pc.repeat(args.cmap_sample_num, 1, 1)
            with torch.no_grad():
                dummy_input = torch.ones((args.cmap_sample_num, args.point_sample_num)).to(device).float()
                recon_h2o_cmap, _ = contact_model.p_sample_loop(data = {'x':dummy_input, 'init_obj_verts': init_grasped_obj_mesh_pcd[:, :, :3], 'goal_obj_verts': goal_grasped_obj_mesh_pcd[:, :, :3], 'scene_pcd': scene_pcd, 'origin_obj_verts': origin_grasped_obj_mesh_pcd})
                recon_h2o_cmap = umnormalize_to_zero_and_one(recon_h2o_cmap)
            
            h2o_obj_pc = obj_pc.clone()
            h2o_obj_pc = torch.cat([h2o_obj_pc, recon_h2o_cmap.unsqueeze(-1).clone()], dim = 2)
            h2o_obj_pc = h2o_obj_pc.repeat(args.grasp_sample_num, 1, 1)
            recon_h2o_cmap = recon_h2o_cmap.repeat(args.grasp_sample_num, 1)
            dummy_input = torch.ones((args.cmap_sample_num * args.grasp_sample_num, 51)).to(device).float()
            with torch.no_grad():
                if args.mode == 'mu':
                    recon_param, _ = grasp_model.p_sample_mu_loop({'x': dummy_input, 'y':h2o_obj_pc.permute(0, 2, 1)})
                elif args.mode == 'noise':
                    recon_param = grasp_model.p_sample_loop({'x': dummy_input, 'y':h2o_obj_pc.permute(0, 2, 1)})
            vertices, keypoints = rh_mano.forward(
                        th_trans=recon_param[:, :3],
                        th_pose_coeffs=recon_param[:, 3:],
                        )

            vertices = vertices / 1000
            d_score = overall_diversity(vertices.detach().cpu().numpy())
            diversity_score.append(d_score)
            obj_diversity_score.append(d_score)
            norm_obj_nn_dist_recon = euclidean_dist(vertices, h2o_obj_pc[:, :, :3])
            h2o_consistency = torch.nn.functional.mse_loss(norm_obj_nn_dist_recon, recon_h2o_cmap)
            h2o_consistency_score.append(h2o_consistency.item())
            obj_h2o_consistency_score.append(h2o_consistency.item())
            repeat_init_grasped_obj_vox = torch.FloatTensor(init_grasped_obj_vox.points).unsqueeze(0).repeat(args.cmap_sample_num * args.grasp_sample_num, 1, 1).cuda()
            init_hand_vertex = vertices + trans
            transform = torch.FloatTensor(init_grasped_pose).unsqueeze(0).repeat(args.cmap_sample_num * args.grasp_sample_num, 1, 1).cuda()
            goal_hand_vertex = torch.bmm(torch.linalg.inv(transform[:, :3, :3]), (init_hand_vertex.permute(0, 2, 1) - transform[:, :3, 3].unsqueeze(-1)))
            transform = torch.FloatTensor(goal_grasped_pose).unsqueeze(0).repeat(args.cmap_sample_num * args.grasp_sample_num, 1, 1).cuda()
            goal_hand_vertex = torch.bmm(transform[:, :3, :3], goal_hand_vertex) + transform[:, :3, 3].unsqueeze(-1)
            goal_hand_vertex = goal_hand_vertex.permute(0, 2, 1)
            volume = gpu_intersect_vox(init_hand_vertex, rh_faces.reshape(-1, 3), repeat_init_grasped_obj_vox, pitch=0.001)
            contactratio = volume > 0
            repeat_total_scene_vertices = scene_mesh.verts_packed().unsqueeze(0).repeat(args.cmap_sample_num * args.grasp_sample_num, 1, 1)
            repeat_total_scene_faces = scene_mesh.faces_packed()
            init_penetration_percentage = kaolin.ops.mesh.check_sign(repeat_total_scene_vertices, repeat_total_scene_faces, init_hand_vertex).sum(dim = 1) / 778
            # print(init_penetration_percentage.shape)
            goal_penetratoin_percentage = kaolin.ops.mesh.check_sign(repeat_total_scene_vertices, repeat_total_scene_faces, goal_hand_vertex).sum(dim = 1) / 778
            # print(init_penetration_percentage.cpu().numpy() < 0.03)
            successful_flag = (init_penetration_percentage.cpu().numpy() < 0.03) & (goal_penetratoin_percentage.cpu().numpy() < 0.03)
            obj_succenssful_ratio.append(successful_flag.sum() / args.grasp_sample_num / args.cmap_sample_num)
            succenssful_ratio.append(successful_flag.sum() / args.grasp_sample_num / args.cmap_sample_num)
            for s in range(args.cmap_sample_num * args.grasp_sample_num):
                # if not successful_flag[s]:
                #     continue
                total_count += 1
                init_grasped_obj_mesh = trimesh.load(os.path.join('models', 'brick', grasped_obj_name, 'textured_simple.obj'))
                init_grasped_obj_mesh.apply_transform(init_grasped_pose)
                hand_mesh = trimesh.Trimesh(vertices = init_hand_vertex[s].detach().cpu().squeeze(0).numpy(), faces=rh_faces.cpu().numpy().reshape((-1, 3)),
                                        face_colors=[int(0.85882353*255), int(0.74117647*255), int(0.65098039*255)])
                penetr_vol = volume[s].item()
                init_penetr_per = init_penetration_percentage[s].item()
                goal_penetr_per = goal_penetratoin_percentage[s].item()
                sample_contact = contactratio[s].item()
                # simulation displacement
                vhacd_exe = "v-hacd/app/build/TestVHACD"
                try:
                    simu_disp = run_simulation(init_hand_vertex[s].detach().cpu().squeeze(0).numpy(), rh_faces.cpu().numpy().reshape((-1, 3)),
                                            init_grasped_obj_mesh.vertices, init_grasped_obj_mesh.faces,
                                            vhacd_exe=vhacd_exe)
                except:
                    simu_disp = 0.10
                    print('vhacd error')
                save_flag = (penetr_vol < args.penetr_vol_thre) and (simu_disp < args.simu_disp_thre) and sample_contact
                print('penetr vol: {}, simu disp: {}, contact: {}, save flag: {}'
                    .format(penetr_vol, simu_disp, sample_contact, save_flag))
                penetration_volume.append(penetr_vol)
                simulation_displacement.append(simu_disp)
                contact_ratio.append(sample_contact)
                obj_penetration_volume.append(penetr_vol)
                obj_simulation_displacement.append(simu_disp)
                obj_contact_ratio.append(sample_contact)
                obj_init_scene_penetration_percentage.append(init_penetr_per)
                obj_goal_scene_penetratoin_percentage.append(goal_penetr_per)
                init_scene_penetration_percentage.append(init_penetr_per)
                goal_scene_penetratoin_percentage.append(goal_penetr_per)
                if penetr_vol < 3e-6 and simu_disp < 0.02:
                    good_ratio.append(1)
                    obj_good_ratio.append(1)
                else:
                    good_ratio.append(0)
                    obj_good_ratio.append(0)
                for index in range(len(qr_threshold)):
                    if penetr_vol < qr_threshold[index][0] and simu_disp < qr_threshold[index][1]:
                        qr[index].append(1)
                    else:
                        qr[index].append(0)
                output['init_pose'].append(init_config)
                output['goal_pose'].append(goal_config)
                output['init_hand_vertex'].append(init_hand_vertex[s].detach().cpu().squeeze(0).numpy())
                output['goal_hand_vertex'].append(goal_hand_vertex[s].detach().cpu().squeeze(0).numpy())
                if args.vis:
                    init_hand_mesh = trimesh.Trimesh(vertices = init_hand_vertex[s].detach().cpu().squeeze(0).numpy(), faces=rh_faces.cpu().numpy().reshape((-1, 3)),
                                        face_colors=[int(0.85882353*255), int(0.74117647*255), int(0.65098039*255)])
                    goal_hand_mesh = trimesh.Trimesh(vertices = goal_hand_vertex[s].detach().cpu().squeeze(0).numpy(), faces=rh_faces.cpu().numpy().reshape((-1, 3)),
                                        face_colors=[int(0.85882353*255), int(0.74117647*255), int(0.65098039*255)])
                    init_grasped_obj_mesh = trimesh.load(os.path.join('models', 'brick', grasped_obj_name, 'textured_simple.obj'))
                    init_grasped_obj_mesh.apply_transform(init_grasped_pose)
                    goal_grasped_obj_mesh = trimesh.load(os.path.join('models', 'brick', grasped_obj_name, 'textured_simple.obj'))
                    goal_grasped_obj_mesh.apply_transform(goal_grasped_pose)
                    cmap=recon_h2o_cmap[s].cpu().numpy()
                    normalized_cmap = (cmap * 255).astype(np.uint8)
                    obj_color = cv2.applyColorMap(normalized_cmap, cv2.COLORMAP_JET)
                    obj_color = obj_color[...,::-1]
                    color = np.full((args.point_sample_num, 4), 255)
                    color[:, :3] = obj_color.reshape(args.point_sample_num, 3)
                    init_grasped_obj_pcd = init_grasped_obj_mesh_pcd[0].cpu().numpy()
                    goal_grasped_obj_pcd = goal_grasped_obj_mesh_pcd[0].cpu().numpy()
                    init_pcd = trimesh.PointCloud(vertices=init_grasped_obj_pcd, colors=color)
                    goal_pcd = trimesh.PointCloud(vertices=goal_grasped_obj_pcd, colors=color)
                    a = trimesh.Scene([init_grasped_obj_mesh] + obs_mesh)
                    a.set_camera(angles=[np.pi/4, 0, 0], center=[0, -0.1, 0.8])
                    png = a.save_image(visible=True)
                    file_name = os.path.join(dirname, 'image', "{}_{}_{}_init_scene.png".format(obj, i, s))
                    with open(file_name, 'wb') as f:
                        f.write(png)
                        f.close()
                    a = trimesh.Scene([goal_grasped_obj_mesh] + obs_mesh)
                    a.set_camera(angles=[np.pi/4, 0, 0], center=[0, -0.1, 0.8])
                    png = a.save_image(visible=True)
                    file_name = os.path.join(dirname, 'image', "{}_{}_{}_goal_scene.png".format(obj, i, s))
                    with open(file_name, 'wb') as f:
                        f.write(png)
                        f.close()

                    a = trimesh.Scene([init_hand_mesh, init_pcd] + obs_mesh)
                    a.set_camera(angles=[np.pi/4, 0, 0], center=[0, -0.1, 0.8])
                    png = a.save_image(visible=True)
                    file_name = os.path.join(dirname, 'image', "{}_{}_{}_init_scene_grasping.png".format(obj, i, s))
                    with open(file_name, 'wb') as f:
                        f.write(png)
                        f.close()

                    a = trimesh.Scene([goal_hand_mesh, goal_pcd] + obs_mesh)
                    a.set_camera(angles=[np.pi/4, 0, 0], center=[0, -0.1, 0.8])
                    png = a.save_image(visible=True)
                    file_name = os.path.join(dirname, 'image', "{}_{}_{}_goal_scene_grasping.png".format(obj, i, s))
                    with open(file_name, 'wb') as f:
                        f.write(png)
                        f.close()

                    a = trimesh.Scene([init_pcd] + obs_mesh)
                    a.set_camera(angles=[np.pi/4, 0, 0], center=[0, -0.1, 0.8])
                    png = a.save_image(visible=True)
                    file_name = os.path.join(dirname, 'image', "{}_{}_{}_init_scene_contact.png".format(obj, i, s))
                    with open(file_name, 'wb') as f:
                        f.write(png)
                        f.close()

                    a = trimesh.Scene([goal_pcd] + obs_mesh)
                    a.set_camera(angles=[np.pi/4, 0, 0], center=[0, -0.1, 0.8])
                    png = a.save_image(visible=True)
                    file_name = os.path.join(dirname, 'image', "{}_{}_{}_goal_scene_contact.png".format(obj, i, s))
                    with open(file_name, 'wb') as f:
                        f.write(png)
                        f.close()
        sample_grasp = args.cmap_sample_num * args.grasp_sample_num
        print('obj: {}'.format(obj), file=file)
        print('Mean h2o-h2o consistency score: {}'.format(np.average(obj_h2o_consistency_score)), file=file)
        print('Mean penetration volume: {}'.format(np.average(obj_penetration_volume)), file=file)
        print('Mean init scene penetration percentage: {}'.format(np.average(obj_init_scene_penetration_percentage)), file=file)
        print('Mean goal scene penetration percentage: {}'.format(np.average(obj_goal_scene_penetratoin_percentage)), file=file)
        print('Mean simulation displacement: {}'.format(np.average(obj_simulation_displacement)), file=file)
        print('Mean diversity score: {:9.5f}'.format(np.average(obj_diversity_score)), file=file)
        print('contact ratio: {}'.format(np.average(obj_contact_ratio)), file=file)
        print('successful ratio: {}'.format(np.average(obj_succenssful_ratio)), file=file)
        print('good ratio: {}'.format(np.average(obj_good_ratio)), file=file)
        print(file=file)
        file.flush()
        np.save(os.path.join(dirname, obj), output)
    
    print('*' * 40, file=file)
    print('Total Mean h2o-h2o consistency score: {}'.format(np.average(h2o_consistency_score)), file=file)
    print('Total Mean penetration volume: {}'.format(np.average(penetration_volume)), file=file)
    print('Total Std penetration volume: {}'.format(np.std(penetration_volume)), file=file)
    print('Total Mean init scene penetration percentage: {}'.format(np.average(init_scene_penetration_percentage)), file=file)
    print('Total Mean goal scene penetration percentage: {}'.format(np.average(goal_scene_penetratoin_percentage)), file=file)
    print('Total Mean simulation displacement: {}'.format(np.average(simulation_displacement)), file=file)
    print('Total Std simulation displacement: {}'.format(np.std(simulation_displacement)), file=file)
    print('Total Mean diversity score: {:9.5f}'.format(np.average(diversity_score)), file=file)
    print('Total contact ratio: {}'.format(np.average(contact_ratio)), file=file)
    print('Total good ratio: {}'.format(np.average(good_ratio)), file=file)
    for index in range(len(qr_threshold)):
        print('Total good ratio of pv={}, sd={}: {}'.format(qr_threshold[index][0], qr_threshold[index][1], np.average(qr[index])), file=file)


        
                
                