import torch
from models.ddnm.guided_diffusion.models import Model
from models.ddnm.guided_diffusion.diffusion import Diffusion
import yaml
import argparse
import torchvision.transforms as T
CONFIG_PATH =  "DDNM/celeba_hq.yml"
WEIGHTS_INPAINT = "DDNM/celeba_hq.ckpt"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_config(config_path, device):
    with open(config_path, "r") as f:
        # Open the config yml file
        config = yaml.safe_load(f)
    
    def dict2namespace(config):
        """Converts dictionary to namespace"""
        namespace = argparse.Namespace()

        for key, val in config.items():
            new_value = dict2namespace(val) \
                        if isinstance(val, dict) else val
            setattr(namespace, key, new_value)

        return namespace
    
    # Convert config to a better form
    config = dict2namespace(config)
    config.device = device
    
    return config

config = load_config(CONFIG_PATH, DEVICE)

class Inpainter():
    def __init__(self) -> None:
        self.inpainter = Model(config)
        # Load the weights to the model and cast to correct device
        self.inpainter.load_state_dict(torch.load(WEIGHTS_INPAINT, map_location=DEVICE))
        self.inpainter.to(DEVICE)
            
    def get_inpainted(self, img, mask):
        class args:
            deg = "inpainting"
            eta = 0.85
            subset_start = -1
            subset_end = -1
            seed = 1234
            sigma_y = 0
        
        # Transform and resize
        transforms = T.Compose([
            T.ToTensor(),
            T.Resize((config.data.image_size, config.data.image_size)),
        ])

        # Pass the image and the mask as configs
        config.image = transforms(img)
        config.mask = transforms(mask != 1)

        # Instantiate the runner and run the code
        runner = Diffusion(args, config, device=DEVICE)
        inpainted = runner.simplified_ddnm_plus(self.inpainter, cls_fn=None)
        
        return inpainted

if __name__ == '__main__':
    Inpainter = Inpainter()
    import torchvision

    import matplotlib.pyplot as plt
    mask = plt.imread('./DDNM/mask.png')
    img = plt.imread('./DDNM/q_512.png')
    mask[mask!=0]=1
    mask[mask!=1]=0
    x = Inpainter.get_inpainted(img, mask)
    torchvision.utils.save_image(x,'./DDNM/saved.png')
    # save_images('./', x)