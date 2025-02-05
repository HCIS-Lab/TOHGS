import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import os
import pytorch3d.ops
from utils import utils_loss
from utils.loss import inter_penetr_loss, contact_inter_penetr_loss
from manopth.manolayer import ManoLayer
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.loss import chamfer_distance
from network.graspdiffusion import UNetModel
from scheduler.ddpm import DDPM
import time
import numpy as np
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


def align_distance(hand, obj):
    obj_normal = pytorch3d.ops.estimate_pointcloud_normals(obj)
    with torch.no_grad():
        batch_object_point_cloud = obj.unsqueeze(1)
        batch_object_point_cloud = batch_object_point_cloud.repeat(1, 778, 1, 1).transpose(1, 2)
    hand_surface_points = hand.unsqueeze(1)
    hand_surface_points = hand_surface_points.repeat(1, obj.size(1), 1, 1)
    with torch.no_grad():
        batch_object_normal_cloud = obj_normal.unsqueeze(1)
        batch_object_normal_cloud = batch_object_normal_cloud.repeat(1, 778, 1, 1).transpose(1, 2)
    object_hand_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
    object_hand_align = ((hand_surface_points - batch_object_point_cloud) *
                            batch_object_normal_cloud).sum(dim=3)
    object_hand_align /= (object_hand_dist + 1e-5)

    object_hand_align_dist = object_hand_dist * torch.exp(2 * (1 - object_hand_align))
    contact_dist = torch.sqrt(object_hand_align_dist.min(dim=2)[0])
    contact_value_current = 1 - 2 * (torch.sigmoid(10 * contact_dist) - 0.5)
    # consistency_loss = (torch.nn.functional.l1_loss(contact_value_current, h2o_cmap.squeeze(1), reduction='none') * (h2o_cmap.squeeze(1))).sum() / h2o_cmap.size(0)
    return contact_value_current


def save(object_pcd, h2o_cmap, gt_hand_pcd, recon_hand_pcd, save_root, epoch, step, mode):
    np.save(os.path.join(save_root, mode, 'object_pcd', "epoch_{}_step_{}.npy".format(epoch, step)), object_pcd)
    np.save(os.path.join(save_root, mode, 'h2o_cmap', "epoch_{}_step_{}.npy".format(epoch, step)), h2o_cmap)
    np.save(os.path.join(save_root, mode, 'gt_hand_pcd', "epoch_{}_step_{}.npy".format(epoch, step)), gt_hand_pcd)
    np.save(os.path.join(save_root, mode, 'recon_hand_pcd', "epoch_{}_step_{}.npy".format(epoch, step)), recon_hand_pcd)


def main(args):
    local_time = time.localtime(time.time())
    time_str = str(local_time[1]) + '_' + str(local_time[2]) + '_' + str(local_time[3]) + '_' + str(local_time[4]) + '_' + str(local_time[5])
    model_root = os.path.join('./logs2/graspdiffusion')
    model_info = 'graspdiffusion_{}_epoch_{}_{}'.format(args.epochs, args.task, time_str)
    save_root = os.path.join(model_root, model_info)
    writer = SummaryWriter('runs/graspdiffusion/{}'.format(model_info))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        os.makedirs(os.path.join(save_root, 'train', 'object_pcd'))
        os.makedirs(os.path.join(save_root, 'train', 'h2o_cmap'))
        os.makedirs(os.path.join(save_root, 'train', 'gt_hand_pcd'))
        os.makedirs(os.path.join(save_root, 'train', 'recon_hand_pcd'))
        os.makedirs(os.path.join(save_root, 'eval', 'object_pcd'))
        os.makedirs(os.path.join(save_root, 'eval', 'h2o_cmap'))
        os.makedirs(os.path.join(save_root, 'eval', 'gt_hand_pcd'))
        os.makedirs(os.path.join(save_root, 'eval', 'recon_hand_pcd'))
    with open(os.path.join(save_root, 'cfg.txt'), '+w') as file:
        print(args, file=file)
    unet = UNetModel().cuda()
    ddpm = DDPM(unet, args.diffusion_step, args.scheduler).float()
    optimizer = torch.optim.AdamW(
        ddpm.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.task == "placing" or args.task == "shelving":
        from dataset.placing_dataset import grasping_pose
    elif args.task == "stacking":
        from dataset.stacking_dataset import grasping_pose

    train_dataset = grasping_pose(mode="train", batch_size=args.batch_size, sample_points=args.sample_num)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.dataloader_workers)
    
    val_dataset = grasping_pose(mode="val", batch_size=args.batch_size, sample_points=args.sample_num)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_loader) * args.epochs) //
        args.gradient_accumulation_steps,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = ddpm.to(device)
    with torch.no_grad():
        manolayer = ManoLayer(mano_root='./models/mano/', flat_hand_mean=True, use_pca=False).to(device=device)
    for epoch in range(args.epochs):
        ddpm.train()
        epoch_total_loss, epoch_recon_loss, epoch_consistency_loss, epoch_diffusion_loss, epoch_penetration_loss = 0, 0, 0, 0, 0
        for step, (data_dcit)in enumerate(train_loader):
            obj_pc, hand_param, hand_pc = data_dcit['obj_verts'].to(device), data_dcit['recon_param'].to(device), data_dcit['hand_verts'].to(device)
            f = manolayer.th_faces.view(1, -1, 3).contiguous()
            rh_faces = f.repeat(obj_pc.size(0), 1, 1)
            obj_normal = pytorch3d.ops.estimate_pointcloud_normals(obj_pc)
            optimizer.zero_grad()
            batch_size = obj_pc.shape[0]
            with torch.no_grad():
                h2o_cmap = euclidean_dist(hand_pc, obj_pc[:, :, :3])
            obj_pc = torch.cat([obj_pc, obj_normal, h2o_cmap.unsqueeze(2)], dim=2).permute(0, 2, 1)
            pred_x0, _, noise = ddpm({'x':hand_param, 'y': obj_pc})
            vertices, _ = manolayer.forward(
                th_trans=pred_x0[:, :3],
                th_pose_coeffs=pred_x0[:, 3:],
                )
            recon_xyz = vertices / 1000
            diffusion_loss = F.mse_loss(pred_x0, hand_param, reduction='sum') / batch_size
            recon_loss = F.mse_loss(recon_xyz, hand_pc, reduction='sum') / batch_size
            recon_h2o_cmap = euclidean_dist(recon_xyz, obj_pc.permute(0, 2, 1)[:, :, :3])
            consistency_loss = (torch.nn.functional.mse_loss(recon_h2o_cmap, h2o_cmap, reduction='none') * (1 + 5 * h2o_cmap)).sum() / batch_size
            obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz)
            penetr_loss = contact_inter_penetr_loss(recon_xyz, rh_faces, obj_pc.permute(0, 2, 1)[:, :, :3],
                                        obj_nn_dist_recon, obj_nn_idx_recon, h2o_cmap)
            loss = args.recon_loss_weight * torch.clamp(recon_loss, 0, 1.0) + args.consistency_loss_weight * consistency_loss \
                + args.penetration_loss_weight * torch.clamp(penetr_loss, 0, 1.0) + args.diffusion_loss_weight * diffusion_loss
            loss.backward()
            if step == len(train_loader) - 1 or step % 10 ==0:
                print("Train Epoch {:02d}/{:02d}, Batch {:04d}/{:d}, Total Loss {:9.5f}, Mesh {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f}, Diffusion loss {:9.5f}".format(
                        epoch, args.epochs, step, len(train_loader) - 1, loss.item(),
                        recon_loss.item(), consistency_loss.item(), penetr_loss.item(), diffusion_loss.item()))
            epoch_total_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_consistency_loss += consistency_loss.item()
            epoch_diffusion_loss += diffusion_loss.item()
            epoch_penetration_loss += penetr_loss.item()
            optimizer.step()
            lr_scheduler.step()
        epoch_total_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        epoch_consistency_loss /= len(train_loader)
        epoch_diffusion_loss /= len(train_loader)
        epoch_penetration_loss /= len(train_loader)
        writer.add_scalars("Training epoch average loss",
                           {'total_loss': epoch_total_loss, 'recon_loss': epoch_recon_loss, 'consistency_loss': epoch_consistency_loss,
                            'diffusion_loss': epoch_diffusion_loss, 'penetration_loss': epoch_penetration_loss},
                            epoch)
        writer.flush()
        if epoch % args.val_interval == 0:
            with torch.no_grad():
                ddpm.eval()
                val_total_loss, val_recon_loss, val_consistency_loss, val_diffusion_loss, val_penetration_loss = 0, 0, 0, 0, 0
                for step, (data_dcit)in enumerate(val_loader):
                    obj_pc, hand_param, hand_pc = data_dcit['obj_verts'].to(device), data_dcit['recon_param'].to(device), data_dcit['hand_verts'].to(device)
                    f = manolayer.th_faces.view(1, -1, 3).contiguous()
                    rh_faces = f.repeat(obj_pc.size(0), 1, 1)
                    obj_normal = pytorch3d.ops.estimate_pointcloud_normals(obj_pc)
                    batch_size = obj_pc.shape[0]
                    h2o_cmap = euclidean_dist(hand_pc, obj_pc[:, :, :3])
                    obj_pc = torch.cat([obj_pc, obj_normal, h2o_cmap.unsqueeze(2)], dim=2).permute(0, 2, 1)
                    pred_x0, _, noise = ddpm({'x':hand_param, 'y': obj_pc})
                    vertices, _ = manolayer.forward(
                        th_trans=pred_x0[:, :3],
                        th_pose_coeffs=pred_x0[:, 3:],
                        )
                    recon_xyz = vertices / 1000
                    diffusion_loss = F.mse_loss(pred_x0, hand_param, reduction='sum') / batch_size
                    recon_loss = F.mse_loss(recon_xyz, hand_pc, reduction='sum') / batch_size
                    recon_h2o_cmap = euclidean_dist(recon_xyz, obj_pc.permute(0, 2, 1)[:, :, :3])
                    consistency_loss = (torch.nn.functional.mse_loss(recon_h2o_cmap, h2o_cmap, reduction='none') * (1 + 5 * h2o_cmap)).sum() / batch_size
                    obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz)
                    penetr_loss = contact_inter_penetr_loss(recon_xyz, rh_faces, obj_pc.permute(0, 2, 1)[:, :, :3],
                                        obj_nn_dist_recon, obj_nn_idx_recon, h2o_cmap)
                    loss = args.recon_loss_weight * torch.clamp(recon_loss, 0, 1.0) + args.consistency_loss_weight * consistency_loss \
                        + args.penetration_loss_weight * torch.clamp(penetr_loss, 0, 1.0) + args.diffusion_loss_weight * diffusion_loss
                    val_total_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_consistency_loss += consistency_loss.item()
                    val_diffusion_loss += diffusion_loss.item()
                    val_penetration_loss += penetr_loss.item()
                val_total_loss /= len(val_loader)
                val_recon_loss /= len(val_loader)
                val_consistency_loss /= len(val_loader)
                val_diffusion_loss /= len(val_loader)
                val_penetration_loss /= len(val_loader)
                writer.add_scalars("Val epoch average loss",
                           {'total_loss': val_total_loss, 'recon_loss': val_recon_loss, 'consistency_loss': val_consistency_loss,
                            'diffusion_loss': val_diffusion_loss, 'penetration_loss': val_penetration_loss},
                            epoch)
                writer.flush()
        if (epoch+1) % args.save_model_epochs == 0:
            with torch.no_grad():
                torch.save(
                    {
                        'model_state': ddpm.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, os.path.join(save_root, 'model_epoch_{}.pth'.format(epoch)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_model_epochs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--dataloader_workers", type = int, default = 32)
    parser.add_argument("--val_interval", type = int, default = 1)
    parser.add_argument("--task", type=str, default='placing')
    parser.add_argument("--diffusion_step", type=int, default=1000)
    parser.add_argument("--scheduler", type=str, default='linear')
    parser.add_argument("--recon_loss_weight", type = float, default=1)
    parser.add_argument("--consistency_loss_weight", type = float, default=0.002)
    parser.add_argument("--penetration_loss_weight", type = float, default=5)
    parser.add_argument("--diffusion_loss_weight", type = float, default=15)
    parser.add_argument("--sample_num", type = int, default= 2048)
    parser.add_argument("--save_train_result_interval", type = int, default=1000)
    parser.add_argument("--save_eval_result_interval", type = int, default=100)
    args = parser.parse_args()

    assert args.scheduler == 'linear' or args.scheduler == 'cos'
    assert args.task == 'stacking' or args.task == 'placing'
    main(args)
   