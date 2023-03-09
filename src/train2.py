import cv2
import torch
import numpy as np
from torch_enhance.models import SRResNet
from torchvision.transforms.functional import to_tensor, to_pil_image

image = cv2.cvtColor(cv2.imread("data/celeba/resized.jpg"), cv2.COLOR_BGR2RGB)

# increase resolution by factor of 2 (e.g. 128x128 -> 256x256)
model = SRResNet(scale_factor=4, channels=3)

x = to_tensor(image).unsqueeze(0)
sr = model(x).squeeze().clip(0, 1)
sr = np.array(to_pil_image(sr))

# sr = cv2.resize(sr, (256, 256), interpolation=cv2.INTER_AREA)
sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
cv2.imwrite("data/celeba/lol.jpg", sr, params=[int(cv2.IMWRITE_JPEG_QUALITY), 95])
