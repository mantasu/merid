import torch
import argparse
import numpy as np

from .models import Model

DEFAULT = {
    "model": {
        "type": "simple",
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16, ],
        "dropout": 0.0,
        "ema_rate": 0.999,
        "ema": True,
        "resamp_with_conv": True,
        "image_size": 256,
        "num_diffusion_timesteps": 1700,
    },
    "diffusion": {
        "var_type": "fixedsmall",
        "num_diffusion_timesteps": 1700,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02
    },
    "time_travel": {
        "T_sampling": 100,
        "travel_length": 1,
        "travel_repeat": 1,
        "num_diffusion_timesteps": 1500,
    }
}

class DDNMInpainter(object):
    def __init__(self, config={}):
        self.eta = config.get("eta", 0.85)
        self.device = config.get("device", "cpu")
        self.model = self._init_model(config.get("model", DEFAULT["model"]))
        self.betas = self._init_betas(config.get("diffusion", DEFAULT["diffusion"]))
        self.skip, self.time_pairs = self._init_time_pairs(config.get("time_travel", DEFAULT["time_travel"]))

    def _init_model(self, model_config):
        # Dummy dict for namespace to comply with models.py fomat
        dummy_dict = {
            "data": {"image_size": model_config.pop("image_size", 256)},
            "diffusion": {"num_diffusion_timesteps": model_config.pop("num_diffusion_timesteps", 1000)},
            "model": model_config,
        }

        # Convert config to namespace and make model
        model_config = self.dict2namespace(dummy_dict)
        model = Model(model_config).to(self.device)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    def _init_betas(self, diffusion_config):
        betas = self.get_beta_schedule(
            beta_schedule=diffusion_config["beta_schedule"],
            beta_start=diffusion_config["beta_start"],
            beta_end=diffusion_config["beta_end"],
            num_diffusion_timesteps=diffusion_config["num_diffusion_timesteps"],
        )
        betas = torch.from_numpy(betas).float().to(self.device)
        return betas
    
    def _init_time_pairs(self, time_travel_config):
        skip = time_travel_config["num_diffusion_timesteps"] //\
               time_travel_config["T_sampling"]

        times = self.get_schedule_jump(
            time_travel_config["T_sampling"],
            time_travel_config["travel_length"],
            time_travel_config["travel_repeat"]
        )
        time_pairs = list(zip(times[:-1], times[1:]))

        return skip, time_pairs

    def dict2namespace(self, config):
        """Converts dictionary to namespace"""
        namespace = argparse.Namespace()

        for key, val in config.items():
            new_value = self.dict2namespace(val) \
                        if isinstance(val, dict) else val
            setattr(namespace, key, new_value)

        return namespace

    def get_beta_schedule(self, beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if beta_schedule == "quad":
            betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  
            betas = 1.0 / np.linspace(
                num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        
        return betas

    def get_schedule_jump(self, T_sampling, travel_length, travel_repeat):
        jumps_range = range(0, T_sampling - travel_length, travel_length)
        jumps = {i: travel_repeat - 1 for i in jumps_range}
        t, ts = T_sampling, []

        while t >= 1:
            t -= 1
            ts.append(t)

            if jumps.get(t, 0) <= 0:
                continue

            jumps[t] -= 1

            for _ in range(travel_length):
                t += 1
                ts.append(t)

        return [*ts, -1]
    
    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    
    def to(self, device):
        self.device = device
        self.model.to(device)
        self.betas = self.betas.to(device)

        return self
    
    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)
    
    def parameters(self):
        return self.model.parameters()

    @torch.no_grad()
    def forward(self, imgs, masks, rand=None):
        x = torch.rand_like(imgs) if rand is None else rand
        n, x0_preds, xs, y = x.size(0), [], [x], imgs * masks
        import tqdm
        
        # reverse diffusion sampling
        for i, j in tqdm.tqdm(self.time_pairs):
            i, j = i * self.skip, -1 if j * self.skip < 0 else j * self.skip

            if j < i: # normal sampling 
                t = torch.ones(n, device=self.device) * i
                next_t  = torch.ones(n, device=self.device) * j
                at = self.compute_alpha(self.betas, t.long())
                at_next = self.compute_alpha(self.betas, next_t.long())
                sigma_t, xt = (1 - at_next ** 2).sqrt(), xs[-1]

                et = self.model(xt, t)

                if et.size(1) == 6:
                    et = et[:, :3]

                # Eq. 12
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                # Eq. 19
                lambda_t = torch.ones_like(at_next)
                gamma_t = torch.ones_like(at_next) * sigma_t ** 3

                # Eq. 17
                x0_t_hat = x0_t - lambda_t * (x0_t * masks - y) * masks

                c1 = (1 - at_next).sqrt() * self.eta
                c2 = (1 - at_next).sqrt() * ((1 - self.eta ** 2) ** 0.5)

                # different from the paper, we use DDIM here instead of DDPM
                xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                x0_preds.append(x0_t)
                xs.append(xt_next)    
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = self.compute_alpha(self.betas, next_t.long())
                xt_next = at_next.sqrt() * x0_preds[-1] + torch.randn_like(x0_preds[-1]) * (1 - at_next).sqrt()
                xs.append(xt_next)

        return xs[-1].clip(-1, 1)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs) 