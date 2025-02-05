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
            # nn.ReLU(True),
        )
        # global feature
        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, self.hparams['feat_dim']),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            # nn.ReLU(True),
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


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = in_channels if out_channels is None else out_channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(1, self.in_channels),
            nn.SiLU(),
            nn.Conv1d(self.in_channels, self.out_channels, 1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.out_channels)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(1, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.out_channels, self.out_channels, 1)
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        h = h + emb_out.unsqueeze(-1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h
    

class PointResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
        obj_emb_channels= 576
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.obj_emb_channels = obj_emb_channels
        self.dropout = dropout
        self.out_channels = in_channels if out_channels is None else out_channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(1, self.in_channels),
            nn.SiLU(),
            nn.Conv1d(self.in_channels, self.out_channels, 1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.out_channels)
        )
        self.obj_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(512 + 512, self.out_channels)
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(1, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.out_channels, self.out_channels, 1)
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, x, emb, obj_feature):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(x.shape)
        # print(obj_feature.shape)
        h = self.in_layers(x) + self.obj_emb_layers(obj_feature).permute(0, 2, 1)
        emb_out = self.emb_layers(emb)
        h = h + emb_out.unsqueeze(-1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            # mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            # mask = repeat(mask, 'b j -> (b h) () j', h=h)
            # print('sim: ', sim.shape)
            # print('mask: ', mask.shape)
            sim.masked_fill_(~mask.repeat(h, 1, 1), max_neg_value)
        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, mult_ff=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, mult=mult_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        # mask = torch.eye(2048, dtype=torch.bool).cuda().unsqueeze(0).repeat(x.size(0), 1, 1)
        x = self.attn1(self.norm1(x), mask = None) + x
        x = self.attn2(self.norm2(x), context=context, mask = None) + x
        x = self.ff(self.norm3(x)) + x
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, mult_ff=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, mult=mult_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context):
        x = self.attn1(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for sequential data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to sequential data.
    """
    def __init__(self, in_channels, n_heads=8, d_head=64,
                 depth=1, dropout=0., context_dim=None, mult_ff=2):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv1d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, mult_ff=mult_ff)
                for d in range(depth)]
        )

        self.proj_out = nn.Conv1d(inner_dim,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        B, C, L,  = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)

        x = rearrange(x, 'b c l -> b l c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b l c -> b c l')
        x = self.proj_out(x)
        return x + x_in


class TaskContactDiffusion(nn.Module):
    def __init__(self) -> None:
        super(TaskContactDiffusion, self).__init__()

        self.d_x = 1
        self.d_model = 256
        self.nblocks = 8
        self.resblock_dropout = 0.
        self.transformer_num_heads = 4
        self.transformer_dim_head = 64
        self.transformer_dropout = 0.1
        self.transformer_depth = 1
        self.transformer_mult_ff = 4
        self.context_dim = 512 + 512
        self.use_position_embedding = False # for input sequence x
        self.time_embed_mult = 4
        ## create scene model from config
        self.scene_model_name = 'pointnet2'
        self.scene_model = PointNet2SemSegSSGShape({'feat_channel':5, 'feat_dim': 512, 'local_size':512}) # 3 for normal
        ## load pretrained weights
        

        time_embed_dim = self.d_model * self.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )
        # self.feat_proj = nn.Linear(512 + 64, self.d_model)
        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                PointResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                    self.context_dim
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(1, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )
        self.silu = nn.SiLU()
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, obj_pc: torch.Tensor) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        # x_t = x_t.reshape(x_t.size(0), -1, self.d_x)
        x_t = x_t.unsqueeze(-1)
        ## condition embedding
        obj_local_feature, obj_global_feature = self.scene_model(obj_pc) #pointnet++
        obj_global_feature = obj_global_feature.unsqueeze(1).repeat(1, obj_local_feature.size(2), 1)
        cond = torch.cat([obj_global_feature, obj_local_feature.permute(0, 2, 1)], dim=2)
        # cond = obj_local_feature.permute(0, 2, 1)

        # cond = self.feat_proj(self.silu(cond))
        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)

        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h) # <B, d_model, L>
        # print(h.shape, cond.shape) # <B, d_model, L>, <B, T , c_dim>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb, cond)
            h = self.layers[i * 2 + 1](h, context=cond)
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        # h = h.reshape(h.size(0), -1)
        h = h.squeeze(-1)
        return h
    def p2p_distance(self, hand, obj, normalized = True, alpha = 100):
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


    
if __name__ == '__main__':
    pass
    # model = ContactGraspDiffusion().cuda()
    # noise_cmap = torch.rand(2, 51).cuda()
    # obj_pc = torch.rand(2, 7, 3000).cuda()
    # timesteps = torch.tensor([2], dtype=torch.long).cuda()
    # noise = model(noise_cmap, timesteps, obj_pc)
    # print(noise.shape)