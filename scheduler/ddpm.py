from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from manopth.manolayer import ManoLayer
from pytorch3d.structures import Meshes
from utils import utils_loss
import numpy as np
from tqdm import tqdm
def euclidean_dist(hand, obj, alpha = 100):
    batch_object_point_cloud = obj.unsqueeze(1)
    batch_object_point_cloud = batch_object_point_cloud.repeat(1, hand.size(1), 1, 1).transpose(1, 2)
    hand_surface_points = hand.unsqueeze(1)
    hand_surface_points = hand_surface_points.repeat(1, obj.size(1), 1, 1)
    object_hand_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
    contact_dist = object_hand_dist.min(dim=2)[0]
    contact_value_current = 1 - 2 * (torch.sigmoid(alpha * contact_dist) - 0.5)
    return contact_value_current


def penetration_loss(hand, obj):
    obj_pcd = obj[:, :, :3]
    obj_nor = obj[:, :, 3:6]
    batch_obj_pcd = obj[:, :, :3].view(obj.size(0), 1, obj.size(1), 3).repeat(1, hand.size(1), 1, 1)
    batch_hand_pcd = hand.view(hand.size(0), hand.size(1), 1, 3).repeat(1, 1, obj.size(1), 1)
    hand_obj_dist = (batch_obj_pcd - batch_hand_pcd).norm(dim=3)
    hand_obj_dist, hand_obj_indices = hand_obj_dist.min(dim=2)
    # gather the obj points and normals w.r.t. hand points
    hand_obj_points = torch.stack([obj_pcd[i, x, :] for i, x in enumerate(hand_obj_indices)], dim=0)
    hand_obj_normals = torch.stack([obj_nor[i, x, :] for i, x in enumerate(hand_obj_indices)], dim=0)
    # compute the signs
    hand_obj_signs = ((hand_obj_points - hand) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    collision_value = (hand_obj_signs * hand_obj_dist).sum(dim=1)
    # loss += collision_value.mean()

    return 100 * collision_value.sum() / hand.size(0)


def consistency_loss(hand_xyz, obj_xyz, cmap):
 
    # cmap consistency loss
    cmap_affordance = euclidean_dist(hand_xyz, obj_xyz)
    consistency_loss = torch.nn.functional.mse_loss(cmap_affordance, cmap, reduction='mean')
    

    
    
    return consistency_loss


def TTT_loss(hand_xyz, hand_face, obj_xyz, cmap):
 
    # cmap consistency loss
    cmap_affordance = euclidean_dist(hand_xyz, obj_xyz)
    consistency_loss = torch.nn.functional.mse_loss(cmap_affordance, cmap)
    # inter-penetration loss
    mesh = Meshes(verts=hand_xyz.cuda(), faces=hand_face.cuda())
    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
    nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
    interior = utils_loss.get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)
    reverse_cmap = 1 - cmap
    penetr_dist = 100 * (nn_dist[interior] * (1 + 3 * reverse_cmap[interior])).mean()
    penetr_dist = torch.nan_to_num(penetr_dist, 0)

    
    
    return penetr_dist, consistency_loss

def make_schedule_ddpm(timesteps = 1000, s=0.008 , beta_schedule='cos') -> Dict:
    if beta_schedule == 'cos':
        x = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    elif beta_schedule == 'linear':
        betas = torch.linspace(0.0001, 0.01, timesteps)
    else:
        raise Exception('Unsupport beta schedule.')

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])    
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod),
        'log_one_minus_alphas_cumprod': torch.log(1 - alphas_cumprod),
        'sqrt_recip_alphas_cumprod': torch.sqrt(1 / alphas_cumprod),
        'sqrt_recipm1_alphas_cumprod': torch.sqrt(1 / alphas_cumprod - 1),
        'posterior_variance': posterior_variance,
        'posterior_log_variance_clipped': torch.log(posterior_variance.clamp(min=1e-20)),
        'posterior_mean_coef1': betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
        'posterior_mean_coef2': (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
    }


class DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, timesteps = 1000, beta_schedule='cos', optimize = False) -> None:
        super(DDPM, self).__init__()
        
        self.eps_model = eps_model
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.optimize = optimize
        self.mano = ManoLayer(mano_root='./models/mano/', flat_hand_mean=True, use_pca=False)
        for k, v in make_schedule_ddpm(timesteps = timesteps, beta_schedule = beta_schedule).items():
            self.register_buffer(k, v)
        
       
        self.criterion = F.mse_loss
                

    @property
    def device(self):
        return self.betas.device
    

    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Forward difussion process, $q(x_t \mid x_0)$, this process is determinative 
        and has no learnable parameters.

        $x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon$

        Args:
            x0: samples at step 0
            t: diffusion step
            noise: Gaussian noise
        
        Return:
            Diffused samples
        """
        B, *x_shape = x0.shape
        x_t = self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise

        return x_t


    def forward(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['x'] gives the target data, data['y'] gives the condition
        
        Return:
            Computed loss
        """
        B = data['x'].shape[0]

        ## randomly sample timesteps
        ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()

        
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        ## apply observation before forwarding to eps model
        ## model need to learn the relationship between the observation frames and the rest frames

        ## predict noise
        cond = data['y']
        pred_noise = self.eps_model(x_t, ts, cond)
        ## apply observation after forwarding to eps model
        ## this operation will detach gradient from the loss of the observation tokens
        ## because the target and output of the observation tokens all are constants


        B, *x_shape = x_t.shape
        pred_x0 = self.sqrt_recip_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * pred_noise
        


        return pred_noise, pred_x0, noise


    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Get and process model prediction

        $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            The predict target `(pred_noise, pred_x0)`, currently we predict the noise, which is as same as DDPM
        """
        B, *x_shape = x_t.shape

        pred_noise = self.eps_model(x_t, t, cond)
        pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise

        return pred_noise, pred_x0


    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Calculate the mean and variance, we adopt the following first equation.

        $\tilde{\mu} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0$
        $\tilde{\mu} = \frac{1}{\sqrt{\alpha}_t}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        B, *x_shape = x_t.shape

        ## predict noise and x0 with model $p_\theta$
        pred_noise, pred_x0 = self.model_predict(x_t, t, cond)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, data: Dict) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
 
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, cond)
        
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0


        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_x


    @torch.no_grad()
    def p_mu_sample(self, x_t: torch.Tensor, t: int, data: Dict) -> torch.Tensor:
        
        B, *x_shape = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
        pred_x0 = self.eps_model(x_t, batch_timestep, cond)
        '''
        #Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        '''
        
        model_mean = self.posterior_mean_coef1[batch_timestep].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[batch_timestep].reshape(B, *((1, ) * len(x_shape))) * x_t
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0
        posterior_variance = self.posterior_variance[batch_timestep].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[batch_timestep].reshape(B, *((1, ) * len(x_shape))) # clipped variance
        with torch.enable_grad():
            if self.optimize and t > 0:
                x_in = model_mean.detach().requires_grad_(True)
                vertices, _ = self.mano(
                    th_trans=x_in[:, :3],
                    th_pose_coeffs=x_in[:, 3:52],
                )
                vertices /= 1000
                # rh_faces = self.mano.th_faces.view(1, -1, 3).contiguous().repeat(x_in.size(0), 1, 1)
                # penetra_loss, consisitency_loss = TTT_loss(vertices, rh_faces, cond.permute(0, 2, 1)[:, :, :3], cond.permute(0, 2, 1)[:, :, -1])
                consisitency_loss = consistency_loss(vertices, cond.permute(0, 2, 1)[:, :, :3], cond.permute(0, 2, 1)[:, :, -1])
                penetra_loss = penetration_loss(vertices, cond.permute(0, 2, 1)[:, :, :6])
                # print(penetra_loss.item())
                loss = 1 * penetra_loss + 0 * consisitency_loss
                grad = torch.autograd.grad(loss, x_in)[0]
                # grad = torch.clip(grad, -1, 1)
                model_mean = model_mean - posterior_variance * grad
        
        
        pred_x = model_mean + (0.5 * posterior_log_variance).exp() * noise

        return pred_x
    
    @torch.no_grad()
    def p_sample_loop(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """
        x_t = torch.randn_like(data['x'], device=self.device)
        ## apply observation to x_t
        
        ## precompute conditional feature, which will be used in every sampling step
        condition = data['y']
        data['cond'] = condition

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data)
            all_x_t.append(x_t)
        return x_t
    
    @torch.no_grad()
    def p_sample_mu_loop(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """
        x_t = torch.randn_like(data['x'], device=self.device)
        ## apply observation to x_t
        
        ## precompute conditional feature, which will be used in every sampling step
        condition = data['y']
        data['cond'] = condition

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_mu_sample(x_t, t, data)
            all_x_t.append(x_t)
        return x_t, all_x_t
    
    @torch.no_grad()
    def sample(self, data: Dict, k: int=1) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        In this method, the sampled results are unnormalized and converted to absolute representation.

        Args:
            data: test data, data['x'] gives the target data shape
            k: the number of sampled data
        
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        ## TODO ddim sample function
        ksamples = []
        for _ in range(k):
            ksamples.append(self.p_sample_loop(data))
        
        ksamples = torch.stack(ksamples, dim=0)
        
      
        return ksamples
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    @torch.no_grad()
    def ddim_sample_mu(
        self,
        data: Dict,
        ddim_timesteps=100,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        sample_img = torch.randn_like(data['x'], device=self.device)
        batch_size = sample_img.shape[0]
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=self.device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=self.device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
    
            # 2. predict noise using model
            pred_x0, _ = self.model_predict(sample_img, t, data['y'])
    

            posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, sample_img.shape) * pred_x0
                + self._extract(self.posterior_mean_coef2, t, sample_img.shape) * sample_img
            )
            posterior_variance = self._extract(self.posterior_variance, t, sample_img.shape)
            posterior_log_variance_clipped = self._extract(
                self.posterior_log_variance_clipped, t, sample_img.shape
            )
            eps  = (self._extract(self.sqrt_recip_alphas_cumprod, t, sample_img.shape) * sample_img
                - pred_x0
            ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, sample_img.shape)
           
            
            sigmas_t = (
                ddim_eta
                * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t))
                * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            noise = torch.randn_like(sample_img)
            mean_pred = (
                pred_x0 * torch.sqrt(alpha_cumprod_t_prev)
                + torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * eps
            )
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(sample_img.shape) - 1)))
            )  # no noise when t == 0
            sample_img = mean_pred + nonzero_mask * sigmas_t * noise

            
        return sample_img
    
class scene_DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, timesteps = 1000, beta_schedule='cos', optimize = False) -> None:
        super(scene_DDPM, self).__init__()
        
        self.eps_model = eps_model
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.optimize = optimize
        self.mano = ManoLayer(mano_root='./models/mano/', flat_hand_mean=True, use_pca=False)
        for k, v in make_schedule_ddpm(timesteps = timesteps, beta_schedule = beta_schedule).items():
            self.register_buffer(k, v)
        
       
        self.criterion = F.mse_loss
                

    @property
    def device(self):
        return self.betas.device
    

    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Forward difussion process, $q(x_t \mid x_0)$, this process is determinative 
        and has no learnable parameters.

        $x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon$

        Args:
            x0: samples at step 0
            t: diffusion step
            noise: Gaussian noise
        
        Return:
            Diffused samples
        """
        B, *x_shape = x0.shape
        x_t = self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise

        return x_t


    def forward(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['x'] gives the target data, data['y'] gives the condition
        
        Return:
            Computed loss
        """
        B = data['x'].shape[0]

        ## randomly sample timesteps
        ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()

        
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        ## apply observation before forwarding to eps model
        ## model need to learn the relationship between the observation frames and the rest frames

        ## predict noise
        init_scene_pc = data['init_scene_pc']
        goal_scene_pc = data['goal_scene_pc']
        pred_noise = self.eps_model(x_t, ts, init_scene_pc, goal_scene_pc)
        ## apply observation after forwarding to eps model
        ## this operation will detach gradient from the loss of the observation tokens
        ## because the target and output of the observation tokens all are constants


        B, *x_shape = x_t.shape
        pred_x0 = self.sqrt_recip_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * pred_noise
        


        return pred_noise, pred_x0, noise


    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, init_scene_pc: torch.Tensor, goal_scene_pc: torch.Tensor) -> Tuple:
        """ Get and process model prediction

        $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            The predict target `(pred_noise, pred_x0)`, currently we predict the noise, which is as same as DDPM
        """
        B, *x_shape = x_t.shape
        pred_noise = self.eps_model(x_t, t, init_scene_pc, goal_scene_pc)
        pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise

        return pred_noise, pred_x0


    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Calculate the mean and variance, we adopt the following first equation.

        $\tilde{\mu} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0$
        $\tilde{\mu} = \frac{1}{\sqrt{\alpha}_t}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        B, *x_shape = x_t.shape

        ## predict noise and x0 with model $p_\theta$
        pred_noise, pred_x0 = self.model_predict(x_t, t, cond)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, data: Dict) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
 
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, cond)
        
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0


        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_x


    @torch.no_grad()
    def p_mu_sample(self, x_t: torch.Tensor, t: int, data: Dict) -> torch.Tensor:
        
        B, *x_shape = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
        init_scene_pc = data['init_scene_pc']
        goal_scene_pc = data['goal_scene_pc']
        pred_noise = self.eps_model(x_t, t, init_scene_pc, goal_scene_pc)
        '''
        #Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        '''
        
        model_mean = self.posterior_mean_coef1[batch_timestep].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[batch_timestep].reshape(B, *((1, ) * len(x_shape))) * x_t
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0
        posterior_variance = self.posterior_variance[batch_timestep].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[batch_timestep].reshape(B, *((1, ) * len(x_shape))) # clipped variance
        with torch.enable_grad():
            if self.optimize and t > 0:
                x_in = model_mean.detach().requires_grad_(True)
                vertices, _ = self.mano(
                    th_trans=x_in[:, :3],
                    th_pose_coeffs=x_in[:, 3:52],
                )
                vertices /= 1000
                # rh_faces = self.mano.th_faces.view(1, -1, 3).contiguous().repeat(x_in.size(0), 1, 1)
                # penetra_loss, consisitency_loss = TTT_loss(vertices, rh_faces, cond.permute(0, 2, 1)[:, :, :3], cond.permute(0, 2, 1)[:, :, -1])
                consisitency_loss = consistency_loss(vertices, cond.permute(0, 2, 1)[:, :, :3], cond.permute(0, 2, 1)[:, :, -1])
                penetra_loss = penetration_loss(vertices, cond.permute(0, 2, 1)[:, :, :6])
                # print(penetra_loss.item())
                loss = 1 * penetra_loss + 0 * consisitency_loss
                grad = torch.autograd.grad(loss, x_in)[0]
                # grad = torch.clip(grad, -1, 1)
                model_mean = model_mean - posterior_variance * grad
        
        
        pred_x = model_mean + (0.5 * posterior_log_variance).exp() * noise

        return pred_x
    
    @torch.no_grad()
    def p_sample_loop(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """
        x_t = torch.randn_like(data['x'], device=self.device)
        ## apply observation to x_t
        
        ## precompute conditional feature, which will be used in every sampling step
        condition = data['y']
        data['cond'] = condition

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data)
            all_x_t.append(x_t)
        return x_t
    
    @torch.no_grad()
    def p_sample_mu_loop(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """
        x_t = torch.randn_like(data['x'], device=self.device)
        ## apply observation to x_t
        
        ## precompute conditional feature, which will be used in every sampling step
        condition = data['y']
        data['cond'] = condition

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_mu_sample(x_t, t, data)
            all_x_t.append(x_t)
        return x_t
    
    @torch.no_grad()
    def sample(self, data: Dict, k: int=1) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        In this method, the sampled results are unnormalized and converted to absolute representation.

        Args:
            data: test data, data['x'] gives the target data shape
            k: the number of sampled data
        
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        ## TODO ddim sample function
        ksamples = []
        for _ in range(k):
            ksamples.append(self.p_sample_loop(data))
        
        ksamples = torch.stack(ksamples, dim=0)
        
      
        return ksamples
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    @torch.no_grad()
    def ddim_sample_mu(
        self,
        data: Dict,
        ddim_timesteps=100,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        sample_img = torch.randn_like(data['x'], device=self.device)
        batch_size = sample_img.shape[0]
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=self.device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=self.device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
    
            # 2. predict noise using model
            pred_x0, _ = self.model_predict(sample_img, t, data['y'])
    

            posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, sample_img.shape) * pred_x0
                + self._extract(self.posterior_mean_coef2, t, sample_img.shape) * sample_img
            )
            posterior_variance = self._extract(self.posterior_variance, t, sample_img.shape)
            posterior_log_variance_clipped = self._extract(
                self.posterior_log_variance_clipped, t, sample_img.shape
            )
            eps  = (self._extract(self.sqrt_recip_alphas_cumprod, t, sample_img.shape) * sample_img
                - pred_x0
            ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, sample_img.shape)
           
            
            sigmas_t = (
                ddim_eta
                * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t))
                * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            noise = torch.randn_like(sample_img)
            mean_pred = (
                pred_x0 * torch.sqrt(alpha_cumprod_t_prev)
                + torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * eps
            )
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(sample_img.shape) - 1)))
            )  # no noise when t == 0
            sample_img = mean_pred + nonzero_mask * sigmas_t * noise

            
        return sample_img