from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch3d.ops
import math
# from network.test import ContactDiffusion
from network.contactdiffusion import TaskContactDiffusion
import numpy as np
from tqdm import tqdm
# import pytorch3d.ops
def make_schedule_ddpm(timesteps = 1000, s=0.008 , beta_schedule='cos') -> Dict:
    if beta_schedule == 'cos':
        x = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    elif beta_schedule == 'linear':
        betas = torch.linspace(0.0001, 0.02, timesteps)
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
    def __init__(self, timesteps = 1000, beta_schedule='cos', alpha = 30) -> None:
        super(DDPM, self).__init__()
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.eps_model = TaskContactDiffusion()
        self.alpha = alpha
        for k, v in make_schedule_ddpm(timesteps = timesteps, beta_schedule = beta_schedule).items():
            self.register_buffer(k, v)
                

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
        scene_pcd = data['scene_pcd']
        init_obj_verts = data['init_obj_verts']
        goal_obj_verts = data['goal_obj_verts']
        origin_obj_verts = data['origin_obj_verts']
        # sample_points, selected_indices = pytorch3d.ops.sample_farthest_points(init_scene_pc.permute(0, 2, 1)[:, :, :3], K = self.K)
        # selected_x_t = data['x'].gather(1, selected_indices.long())
        ## randomly sample timesteps
        ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        ## apply observation before forwarding to eps model
        ## model need to learn the relationship between the observation frames and the rest frames

        ## predict noise
        
        with torch.no_grad():
            init_o2o_cmap = self.eps_model.p2p_distance(scene_pcd, init_obj_verts[:, :, :3], normalized=True, alpha=self.alpha).float()
            goal_o2o_cmap = self.eps_model.p2p_distance(scene_pcd, goal_obj_verts[:, :, :3], normalized=True, alpha=self.alpha).float()
            origin_obj_normal = pytorch3d.ops.estimate_pointcloud_normals(origin_obj_verts)
            o2o_obj_pc = torch.cat([origin_obj_verts, origin_obj_normal, init_o2o_cmap.unsqueeze(-1), goal_o2o_cmap.unsqueeze(-1)], dim = 2)
        pred_noise = self.eps_model(x_t, ts, o2o_obj_pc.permute(0, 2, 1))
        ## apply observation after forwarding to eps model
        ## this operation will detach gradient from the loss of the observation tokens
        ## because the target and output of the observation tokens all are constants


        B, *x_shape = x_t.shape
        pred_x0 = self.sqrt_recip_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * pred_noise
        return pred_noise, pred_x0, noise
    
    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, data: torch.Tensor) -> Tuple:
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
        scene_pcd = data['scene_pcd']
        init_obj_verts = data['init_obj_verts']
        goal_obj_verts = data['goal_obj_verts']
        origin_obj_verts = data['origin_obj_verts']
        with torch.no_grad():
            init_o2o_cmap = self.eps_model.p2p_distance(scene_pcd, init_obj_verts[:, :, :3], normalized=True, alpha=self.alpha).float()
            goal_o2o_cmap = self.eps_model.p2p_distance(scene_pcd, goal_obj_verts[:, :, :3], normalized=True, alpha=self.alpha).float()
            origin_obj_normal = pytorch3d.ops.estimate_pointcloud_normals(origin_obj_verts)
            o2o_obj_pc = torch.cat([origin_obj_verts, origin_obj_normal, init_o2o_cmap.unsqueeze(-1), goal_o2o_cmap.unsqueeze(-1)], dim = 2)
        
        pred_noise = self.eps_model(x_t, t, o2o_obj_pc.permute(0, 2, 1))
        pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise

        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, data: torch.Tensor) -> Tuple:
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
        pred_noise, pred_x0 = self.model_predict(x_t, t, data)

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

        # if 'cond' in data:
        #     ## use precomputed conditional feature
        #     cond = data['cond']
        # else:
        #     ## recompute conditional feature every sampling step
        #     cond = self.eps_model.condition(data)
 
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, data)
        model_mean = torch.clamp(model_mean, -1, 1)
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
        # condition = data['y']
        # data['cond'] = condition

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data)
            all_x_t.append(x_t)
        return x_t, all_x_t
    
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
    def ddim_sample(
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
            pred_noise, _ = self.model_predict(sample_img, t, data)
            
            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev
            
        return sample_img
    
   