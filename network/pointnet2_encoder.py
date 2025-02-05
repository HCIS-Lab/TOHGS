from typing import Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import math
from torch import einsum
from einops import repeat, rearrange
from inspect import isfunction
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class LetentEncoder(nn.Module):
    def __init__(self, in_dim, dim, out_dim):
        super().__init__()
        self.block = ResnetBlockFC(size_in=in_dim, size_out=dim, size_h=dim)
        self.fc_mean = nn.Linear(dim, out_dim)
        self.fc_logvar = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.block(x, final_nl=True)
        return self.fc_mean(x), self.fc_logvar(x)


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, final_nl=False):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        x_out = x_s + dx
        if final_nl:
            return F.leaky_relu(x_out, negative_slope=0.2)
        return x_out


class Pointnet(nn.Module):
    ''' PointNet-based encoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=3):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(hidden_dim, 2 * hidden_dim, 1)
        self.conv3 = torch.nn.Conv1d(2 * hidden_dim, 4 * hidden_dim, 1)
        self.conv4 = torch.nn.Conv1d(4 * hidden_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn3 = nn.BatchNorm1d(4 * hidden_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # x = x.permute(0, 2, 1)

        return x, self.pool(x, dim=1)


class PointNet2SemSegSSGShape(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[self.hparams['feat_channel'], 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 256, 256],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + self.hparams['feat_channel'], 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 256, 256, 256]))
        # local feature
        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['local_size'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['local_size']),
            nn.ReLU(True),
        )
        # global feature
        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, self.hparams['feat_dim']),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        pointcloud = pointcloud.permute(0, 2, 1)
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        bottleneck_feats = l_features[-1].squeeze(-1)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0]), self.fc_layer2(bottleneck_feats)


class autoencoder(nn.Module):
    def __init__(self) -> None:
        super(autoencoder, self).__init__()
        self.decoder = Pointnet(in_dim=512 + 64, out_dim=6)
        self.encoder = PointNet2SemSegSSGShape({'feat_channel':3, 'feat_dim': 512, 'local_size':64}) # 3 for normal
        
    def forward(self, obj_pc) -> torch.Tensor:
        local_feature, global_feature = self.encoder(obj_pc)
        global_feature = global_feature.unsqueeze(-1).repeat(1,1,obj_pc.size(2))
        obj_feature = torch.cat([local_feature, global_feature], dim=1)
        recon_obj_pc, _ = self.decoder(obj_feature)
        
        return recon_obj_pc
    

class VAE(nn.Module):
    def __init__(self) -> None:
        super(VAE, self).__init__()
        self.obj_encoder = PointNet2SemSegSSGShape({'feat_channel':4, 'feat_dim': 4096, 'local_size': 128}) # 3 for normal
        self.encoder = nn.Sequential(
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,),
            nn.ReLU(),
        )
        self.latent_encoder = LetentEncoder(in_dim=256, dim = 128, out_dim=64)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, obj_pc):
        B = obj_pc.size(0)
        N = obj_pc.size(2)
        _, global_feature = self.obj_encoder(obj_pc)
        h = self.encoder(global_feature)
        mu, log_var = self.latent_encoder(h)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=mu.device)
        z = eps * std + mu
        recon_obj_pc = self.decoder(z)
        recon_obj_pc = self.sigmoid(recon_obj_pc)
        return recon_obj_pc.reshape(B, N), mu, log_var, z

PointNet2SemSegSSGShape({'feat_channel':3, 'feat_dim': 2048, 'local_size': 64})
class ContactNet4(nn.Module):
    def __init__(self):
        super(ContactNet4, self).__init__()
        self.n_neurons = 256
        self.latentD = 16
        self.hc = 64
        self.object_feature = 7
        encode_dim = self.hc
        self.obj_pointnet = PointNet2SemSegSSGShape({'feat_channel':3, 'feat_dim': 2048, 'local_size': 64})
        self.h2o_encoder = Pointnet(in_dim=encode_dim + 2, hidden_dim=self.hc, out_dim=self.hc)
        self.o2o_encoder = Pointnet(in_dim=encode_dim + 1, hidden_dim=self.hc, out_dim=self.latentD)
        self.h2o_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.h2o_decoder = Pointnet(in_dim=encode_dim + self.latentD + self.latentD, hidden_dim=self.hc, out_dim=1)

    def forward(self, obj_pc, h2o_cmap, o2o_cmap):
        obj_cond = self.obj_pointnet(obj_pc)
        _, h2o_latent = self.h2o_encoder(torch.cat([obj_cond, o2o_cmap, h2o_cmap], -1))
        _, o2o_latent = self.o2o_encoder(torch.cat([obj_cond, o2o_cmap], -1))
        h2o_mu, h2o_std = self.h2o_latent(h2o_latent)
        z_contact = torch.distributions.normal.Normal(h2o_mu, torch.exp(h2o_std))
        z_s_contact = z_contact.rsample()
        z_s_contact = z_s_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        o2o_latent = o2o_latent.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        contacts_object, _ = self.h2o_decoder(torch.cat([z_s_contact, obj_cond, o2o_latent], -1))
        contacts_object = torch.sigmoid(contacts_object)

        return contacts_object, h2o_mu, h2o_std


if __name__ == '__main__':
    model = autoencoder().cuda()
    obj_pc = torch.rand(2, 6, 1024).cuda()
    timesteps = torch.tensor([2], dtype=torch.long).cuda()
    recon_obj_pcd = model(obj_pc)
    print(recon_obj_pcd.shape)