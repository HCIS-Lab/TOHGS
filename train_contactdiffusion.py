import argparse
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
import os
import pytorch3d.ops
from torch.utils.tensorboard import SummaryWriter
from scheduler.ddpm_task_contact import DDPM
import time
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
from kaolin.metrics.trianglemesh import point_to_mesh_distance
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

def normalized_dis(contact_dist, alpha = 30):
    contact_value_current = 1 - 2 * (torch.sigmoid(alpha * contact_dist) - 0.5)
    return contact_value_current

def normalize_to_minus_one_and_one(img):
    return img * 2 - 1

def umnormalize_to_zero_and_one(img):
    return (img + 1) / 2

def save(init_object_pcd, recon_h2o_cmap, gt_h2o_cmap, save_root, epoch, step, mode):
    np.save(os.path.join(save_root, mode, 'init_object_pcd', "epoch_{}_step_{}.npy".format(epoch, step)), init_object_pcd)
    np.save(os.path.join(save_root, mode, 'recon_h2o_cmap', "epoch_{}_step_{}.npy".format(epoch, step)), recon_h2o_cmap)
    np.save(os.path.join(save_root, mode, 'gt_h2o_cmap', "epoch_{}_step_{}.npy".format(epoch, step)), gt_h2o_cmap)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_time = time.localtime(time.time())
    time_str = str(local_time[1]) + '_' + str(local_time[2]) + '_' + str(local_time[3]) + '_' + str(local_time[4]) + '_' + str(local_time[5])
    model_root = os.path.join('./logs2/contactdiffusion')
    model_info = 'contactdiffusion_{}_epoch_{}_{}'.format(args.num_epochs, args.task, time_str)
    save_root = os.path.join(model_root, model_info)
    writer = SummaryWriter('runs/contactdiffusion/{}'.format(model_info))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        os.makedirs(os.path.join(save_root, 'eval', 'init_object_pcd'))
        os.makedirs(os.path.join(save_root, 'eval', 'recon_h2o_cmap'))
        os.makedirs(os.path.join(save_root, 'eval', 'gt_h2o_cmap'))
        os.makedirs(os.path.join(save_root, 'train', 'init_object_pcd'))
        os.makedirs(os.path.join(save_root, 'train', 'recon_h2o_cmap'))
        os.makedirs(os.path.join(save_root, 'train', 'gt_h2o_cmap'))
    with open(os.path.join(save_root, 'cfg.txt'), '+w') as file:
        print(args, file=file)
    ddpm = DDPM(args.diffusion_step, args.scheduler, alpha=args.alpha).float().cuda()
    ddpm = torch.nn.DataParallel(ddpm.cuda())
    optimizer = torch.optim.AdamW(
        ddpm.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.task == "placing":
        from dataset.placing_dataset import scene
    elif args.task == "stacking":
        from dataset.stacking_dataset import scene
    elif args.task == 'shelving':
        from dataset.shelving_dataset import scene

    train_dataset = scene(mode="train", batch_size=args.batch_size, sample_points=args.sample_num)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.dataloader_workers)
    
    val_dataset = scene(mode="val", batch_size=args.batch_size, sample_points=args.sample_num)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_loader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )    
    for epoch in range(args.num_epochs):
        ddpm.train()
        epoch_total_loss, epoch_diffusion_loss = 0, 0
        for step, (data_dcit)in enumerate(train_loader):
            optimizer.zero_grad()
            hand_pcd, init_obj_verts, goal_obj_verts, origin_obj_verts, scene_pcd = data_dcit['init_hand_verts'].to(device), data_dcit['init_obj_verts'].to(device), data_dcit['goal_obj_verts'].to(device), data_dcit['origin_obj_verts'].to(device), data_dcit['scene_pc'].to(device)
            with torch.no_grad():
                h2o_cmap = euclidean_dist(hand_pcd, init_obj_verts)
                h2o_cmap = h2o_cmap * 2 - 1
                origin_obj_normal = pytorch3d.ops.estimate_pointcloud_normals(origin_obj_verts)
            pred_noise, _, noise = ddpm({'x':h2o_cmap, 'origin_obj_verts': origin_obj_verts, 'init_obj_verts': init_obj_verts, 'goal_obj_verts': goal_obj_verts, 'scene_pcd':scene_pcd, 'origin_obj_normal':origin_obj_normal})
            diffusion_loss = F.mse_loss(pred_noise, noise, reduction='mean')
            loss = diffusion_loss
            loss.backward()
            if args.use_clip_grad:
                clip_grad_value_(ddpm.parameters(), 1.0)
            
            if step == len(train_loader) - 1 or step % 10 ==0:
                print("Train Epoch {:02d}/{:02d}, Batch {:04d}/{:d}, Total Loss {:9.5f}, Diffusion loss {:9.5f}".format(
                        epoch, args.num_epochs, step, len(train_loader) - 1, loss.item(),
                        diffusion_loss.item()))
            epoch_total_loss += loss.item()
            epoch_diffusion_loss += diffusion_loss.item()
            optimizer.step()
            lr_scheduler.step()
            
            
        epoch_total_loss /= len(train_loader)
        epoch_diffusion_loss /= len(train_loader)
        writer.add_scalars("Training epoch average loss",
                           {'total_loss': epoch_total_loss, 'diffusion_loss': epoch_diffusion_loss},
                            epoch)
        writer.flush()
        with torch.no_grad():
            ddpm.eval()
            val_total_loss, val_diffusion_loss = 0, 0
            for step, (data_dcit)in enumerate(val_loader):
                hand_pcd, init_obj_verts, goal_obj_verts, origin_obj_verts, scene_pcd = data_dcit['init_hand_verts'].to(device), data_dcit['init_obj_verts'].to(device), data_dcit['goal_obj_verts'].to(device), data_dcit['origin_obj_verts'].to(device), data_dcit['scene_pc'].to(device)
                with torch.no_grad():
                    h2o_cmap = euclidean_dist(hand_pcd, init_obj_verts)
                    h2o_cmap = h2o_cmap * 2 - 1
                    origin_obj_normal = pytorch3d.ops.estimate_pointcloud_normals(origin_obj_verts)
                pred_noise, _, noise = ddpm({'x':h2o_cmap, 'origin_obj_verts': origin_obj_verts, 'init_obj_verts': init_obj_verts, 'goal_obj_verts': goal_obj_verts, 'scene_pcd':scene_pcd, 'origin_obj_normal':origin_obj_normal})
                diffusion_loss = F.mse_loss(pred_noise, noise, reduction='mean')
                loss = diffusion_loss
                val_total_loss += loss.item()
                val_diffusion_loss += diffusion_loss.item()
            val_total_loss /= len(val_loader)
            val_diffusion_loss /= len(val_loader)
            writer.add_scalars("Val epoch average loss",
                            {'total_loss': val_total_loss, 'diffusion_loss': val_diffusion_loss},
                                epoch)
            writer.flush()
        if (epoch+1) % args.save_model_epochs == 0:
            with torch.no_grad():
                torch.save(
                    {
                        'model_state': ddpm.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'lr_scheduler_state': lr_scheduler.state_dict(),
                        'epoch': epoch 
                    }, os.path.join(save_root, 'model_epoch_{}.pth'.format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Simple example of a training script.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_model_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--dataloader_workers", type = int, default = 90)
    parser.add_argument("--val_interval", type = int, default = 1)
    parser.add_argument("--task", type=str, default='stacking')
    parser.add_argument("--diffusion_step", type=int, default=1000)
    parser.add_argument('--max_grad_value', type=float, default=1)
    parser.add_argument("--scheduler", type=str, default='linear')
    parser.add_argument("--sample_num", type = int, default= 1024)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument("--K", type = int, default = 128)
    parser.add_argument("--pe", default=False, action='store_true')
    parser.add_argument("--alpha", type = int, default = 50)
    parser.add_argument("--save_train_result_interval", type = int, default=1000)
    parser.add_argument('--normalized', default=False, action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    main(args)