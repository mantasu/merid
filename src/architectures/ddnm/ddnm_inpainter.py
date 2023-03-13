import tqdm
import torch
import argparse
import numpy as np
import torch.nn as nn

from .models import Model
from typing import Iterable, Any


class DDNMInpainter(object):
    def __init__(self,
                 num_diff_steps: int = 1000,
                 image_size: int = 256,
                 eta: float = 0.85,
                 device: str = "cpu", 
                 config_model: dict[str, Any] = {},
                 config_diffusion: dict[str, Any] = {},
                 config_time_travel: dict[str, Any] = {}):
        # Assign basics
        self.eta = eta
        self.device = device

        # Acquire specific diffusion attributes like model and schedules
        self.model = self._init_model(config_model, image_size, num_diff_steps)
        self.betas = self._init_betas(config_diffusion, num_diff_steps)
        out = self._init_time_pairs(config_time_travel, num_diff_steps)
        self.skip, self.time_pairs = out

    def _init_model(self, model_config: dict[str, Any], image_size: int,
                    num_diff_steps: int) -> nn.Module:
        # Default model configuration
        DEFAULT_MODEL_CONFIG = {
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
        }
        # Dummy dict for namespace to comply with models.py fomat
        dummy_dict = {
            "data": {"image_size": image_size},
            "diffusion": {"num_diffusion_timesteps": num_diff_steps},
            "model": dict(DEFAULT_MODEL_CONFIG, **model_config),
        }

        # Convert config to namespace and make model
        model_config = self.dict2namespace(dummy_dict)
        model = Model(model_config).to(self.device)
        
        return model
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs) 

    def _init_betas(self, diffusion_config: dict[str, Any],
                    num_diff_steps: int) -> torch.Tensor:
        # Get beta schedule parameter
        betas = self.get_beta_schedule(
            beta_schedule=diffusion_config.get("beta_schedule", "linear"),
            beta_start=diffusion_config.get("beta_start", 0.0001),
            beta_end=diffusion_config.get("beta_end", 0.02),
            num_diff_steps=num_diff_steps,
        )
        # Convert beta schedule parameter to torch tensor
        betas = torch.from_numpy(betas).float().to(self.device)

        return betas
    
    def _init_time_pairs(self, time_travel_config: dict[str, Any],
                         num_diff_steps: int) \
                         -> tuple[int, list[tuple[int, int]]]:
        # Compute the time intervals of skipping (based on T sampling freq)
        skip = num_diff_steps // time_travel_config.get("T_sampling", 100)

        # Calculate times for schedule jumps
        times = self.get_schedule_jump(
            time_travel_config.get("T_sampling", 100),
            time_travel_config.get("travel_length", 1),
            time_travel_config.get("travel_repeat", 1)
        )

        # Generate a list of schedule jump pairs
        time_pairs = list(zip(times[:-1], times[1:]))

        return skip, time_pairs

    def dict2namespace(self, config: dict[str, Any]) -> argparse.Namespace:
        """Converts dictionary to namespace"""

        # Get the namespace form argparse
        namespace = argparse.Namespace()

        for key, val in config.items():
            # Convert sub-dicts to sub-namespaces
            new_value = self.dict2namespace(val) \
                        if isinstance(val, dict) else val
            
            # Set the attribute for the namespace
            setattr(namespace, key, new_value)

        return namespace

    def get_beta_schedule(self, beta_schedule: str, *, beta_start: float,
                          beta_end: float, num_diff_steps: int) -> np.ndarray:
        def sigmoid(x: np.ndarray) -> np.ndarray:
            # Helper sigmoid function
            return 1 / (np.exp(-x) + 1)

        if beta_schedule == "quad":
            # Generate betas in quad linspace
            betas = np.linspace(beta_start ** 0.5,
                                beta_end ** 0.5,
                                num_diff_steps,
                                dtype=np.float64) ** 2

        elif beta_schedule == "linear":
            # Generate betas in linear space
            betas = np.linspace(beta_start,
                                beta_end,
                                num_diff_steps,
                                dtype=np.float64)
        elif beta_schedule == "const":
            # Generate constant size betas based on beta_end
            betas = beta_end * np.ones(num_diff_steps,
                                       dtype=np.float64)
        elif beta_schedule == "jsd":
            # Generate betas in jsd linear space
            betas = 1.0 / np.linspace(num_diff_steps,
                                      1,
                                      num_diff_steps,
                                      dtype=np.float64)
        elif beta_schedule == "sigmoid":
            # Generate betas in sigmoid affected linspace
            betas = np.linspace(-6, 6, num_diff_steps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        
        # Ensure betas shape equals num diff steps
        assert betas.shape == (num_diff_steps,)
        
        return betas

    def get_schedule_jump(self, T_sampling: int, travel_length: int,
                          travel_repeat: int) -> list[int]:
        # Create a jumps range, jumps dictionary and initialize ts list
        jumps_range = range(0, T_sampling - travel_length, travel_length)
        jumps = {i: travel_repeat - 1 for i in jumps_range}
        t, ts = T_sampling, []

        while t >= 1:
            # Update ts
            t -= 1
            ts.append(t)

            if jumps.get(t, 0) <= 0:
                # Skip this jump
                continue
            
            # Subtract one
            jumps[t] -= 1

            for _ in range(travel_length):
                # Update ts
                t += 1
                ts.append(t)

        return [*ts, -1]
    
    def compute_alpha(self, beta: torch.Tensor, t: int) -> torch.Tensor:
        # Concatinate zeros to beta and 
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        
        return a
    
    def to(self, device: str):
        # Everything to device
        self.device = device
        self.model.to(device)
        self.betas = self.betas.to(device)

        return self
    
    def eval(self):
        # Set to eval mode
        self.model.eval()
    
    def load_state_dict(self, weights):
        # Load weights for inner model
        self.model.load_state_dict(weights)
    
    def parameters(self) -> Iterable:
        # Just return model parmaeters
        return self.model.parameters()
    
    def freeze(self):
        for param in self.parameters():
            # Disable inner model grads
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor, masks: torch.Tensor,
                rand: torch.Tensor | None = None, show_progress: bool = False) \
                -> torch.Tensor:
        x = torch.rand_like(imgs) if rand is None else rand
        n, x0_preds, xs, y = x.size(0), [], [x], imgs * masks
        
        pbar = tqdm.tqdm(self.time_pairs) if show_progress else self.time_pairs
        
        # reverse diffusion sampling
        for i, j in pbar:
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
                xt_next = at_next.sqrt() * x0_t_hat + gamma_t * \
                          (c1 * torch.randn_like(x0_t) + c2 * et)

                x0_preds.append(x0_t)
                xs.append(xt_next)    
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = self.compute_alpha(self.betas, next_t.long())
                xt_next = at_next.sqrt() * x0_preds[-1] + \
                          torch.randn_like(x0_preds[-1]) * (1 - at_next).sqrt()
                xs.append(xt_next)

        return xs[-1].clip(-1, 1)