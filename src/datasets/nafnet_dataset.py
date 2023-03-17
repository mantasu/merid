import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torchvision.transforms.functional import normalize
from torch.utils.data import Subset

from albumentations.pytorch.transforms import img_to_tensor, mask_to_tensor

import sys
sys.path.append("src")

from utils.io import load_image
from utils.augment import create_default_augmentation, unnormalize
from PIL import Image

class NafnetDataset(Dataset):
    def __init__(self, syn_path="data/synthetic", celeba_path="data/celeba", target="train", transform = None):
        super().__init__()

        samples = []

        self.transform = A.Compose([], additional_targets={"no_glasses": "image", "inpainted": "image"}) if transform is None else transform

        root_syn = os.path.join(syn_path, target)
        # root_cel = os.path.join(celeba_path, target)
        

        # for file in os.listdir(os.path.join(root_cel, "glasses")):
        #     img_glas = os.path.join(root_cel, "glasses", file)
        #     img_inp_full = os.path.join(root_cel, "intermediate", file[:-4] + "-gen-inpainted-full.jpg")
        #     img_inp_frame = os.path.join(root_cel, "intermediate", file[:-4] + "-gen-inpainted-frame.jpg")
        #     mask_gen_full = os.path.join(root_cel, "intermediate", file[:-4] + "-gen-mask-full.jpg")
        #     mask_gen_frame = os.path.join(root_cel, "intermediate", file[:-4] + "-gen-mask-frame.jpg")

        #     samples.append([img_glas, img_inp_frame, mask_gen_frame, False])
        #     samples.append([img_glas, img_inp_full, mask_gen_full, False])
        
        # for i, file in enumerate(sorted(list(os.listdir(os.path.join(root_cel, "no_glasses"))))[:len(samples)]):
        #     img_no_glas = os.path.join(root_cel, "no_glasses", file)
        #     samples[i].insert(1, img_no_glas)
        

        for file in os.listdir(os.path.join(root_syn, "glasses")):
            
            img_glas = os.path.join(root_syn, "glasses", file)
            img_sun_glass = os.path.join(root_syn, "sunglasses", file.replace("-all", "-sunglasses"))
            img_no_glass = os.path.join(root_syn, "no_glasses", file.replace("-all", "-face"))
            img_inp_full = os.path.join(root_syn, "generated", file[:-4] + "-gen-inpainted-full.jpg")
            img_inp_frame = os.path.join(root_syn, "generated", file[:-4] + "-gen-inpainted-frame.jpg")
            mask_gen_full = os.path.join(root_syn, "generated", file[:-4] + "-gen-mask-full.jpg")
            mask_gen_frame = os.path.join(root_syn, "generated", file[:-4] + "-gen-mask-frame.jpg")

            samples.append([img_glas, img_no_glass, img_inp_frame, mask_gen_frame])
            samples.append([img_sun_glass, img_no_glass, img_inp_full, mask_gen_full])
        
        random.seed(0)
        random.shuffle(samples)

        self.samples = samples
    
    def load_sample(self, sample):
        # Note: NO NORMALIZATION
        samples = self.transform(image=load_image(sample[0]), no_glasses=load_image(sample[1]), inpainted=load_image(sample[2]), mask=load_image(sample[3], convert_to_grayscale=True))
        samples["image"] = img_to_tensor(samples["image"], normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
        samples["inpainted"] = img_to_tensor(samples["inpainted"], normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
        samples["no_glasses"] = img_to_tensor(samples["no_glasses"], normalize=None)
        samples["mask"] = mask_to_tensor(samples["mask"], num_classes=1, sigmoid=None)

        # samples["no_glasses"] = unnormalize(samples["no_glasses"])
        # print(samples["image"].shape, samples["image"].dtype, samples["image"].min(), samples["image"].max())
        # print(samples["inpainted"].shape, samples["inpainted"].dtype, samples["inpainted"].min(), samples["inpainted"].max())
        # print(samples["no_glasses"].shape, samples["no_glasses"].dtype, samples["no_glasses"].min(), samples["no_glasses"].max())
        # print(samples["mask"].shape, samples["mask"].dtype, samples["mask"].min(), samples["mask"].max(), samples["mask"][samples["mask"] > 0].mean())
        # samples["image"] = normalize(samples["image"], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # samples["inpainted"] = normalize(samples["inpainted"], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # print("HERE")

        # import matplotlib.pyplot as plt
        # import numpy as np
        # print(samples["mask"].shape, sample[3])
        # im = load_image(sample[3])#, convert_to_grayscale=True)[..., None].repeat(3, axis=2)
        # # my_im = (samples["mask"] * 255)[..., None].repeat(1, 1, 3).numpy().astype(np.uint8)
        # # print(samples["mask"].shape)
        # # msk = np.moveaxis((samples["mask"] * 255).numpy(), 0, -1).repeat(3, axis=2)
        # msk = samples["mask"].numpy()
        # print(msk.shape)
        # plt.imshow(msk)
        # plt.show()

        # print(samples.keys())

        return tuple(samples.values())

    def __getitem__(self, index):
        return self.load_sample(self.samples[index])
    
    def __len__(self):
        return len(self.samples)


class NafnetGANDataset(Dataset):
    def __init__(self, target="train", transform = None):
        super().__init__()
        samples = []
        self.transform = A.Compose([A.Normalize(), ToTensorV2()], additional_targets={"no_glasses": "image", "inpainted": "image"}) if transform is None else transform
        root_cel = os.path.join("data/celeba", target)
        

        for file in os.listdir(os.path.join(root_cel, "glasses")):
            img_glas = os.path.join(root_cel, "glasses", file)
            img_inp = os.path.join(root_cel, "generated", file[:-4] + "-gen-inpainted.jpg")
            mask_gen = os.path.join(root_cel, "generated", file[:-4] + "-gen-mask.jpg")

            samples.append([img_glas, img_inp, mask_gen])
        
        for i, file in enumerate(sorted(list(os.listdir(os.path.join(root_cel, "no_glasses"))))[:len(samples)]):
            img_no_glas = os.path.join(root_cel, "no_glasses", file)
            samples[i].insert(1, img_no_glas)
        
        random.seed(0)
        random.shuffle(samples)

        self.samples = samples
    
    def load_sample(self, sample):
        samples = self.transform(image=load_image(sample[0]), no_glasses=load_image(sample[1]), inpainted=load_image(sample[2]), mask=load_image(sample[3], convert_to_grayscale=True))
        samples["no_glasses"] = unnormalize(samples["no_glasses"])

        return tuple(samples.values())

    def __getitem__(self, index):
        return self.load_sample(self.samples[index])
    
    def __len__(self):
        return len(self.samples)




class NafnetDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        # Assign dataset attributes
        self.train_transform = None
        self.loader_kwargs = {}

        
        # Create a default augmentation composition
        self.train_transform = create_default_augmentation()

        # Set some default data loader arguments
        self.loader_kwargs.setdefault("batch_size", 8)
        self.loader_kwargs.setdefault("num_workers", 8)
        self.loader_kwargs.setdefault("pin_memory", True)

    def train_dataloader(self) -> DataLoader:
        # Create train dataset and return loader
        train_dataset = NafnetDataset(
            target="train",
            transform=self.train_transform
        )

        train_dataset2 = NafnetGANDataset(
            target="train",
            transform=self.train_transform,
        )
        return {"synthetic": DataLoader(train_dataset, shuffle=True, **self.loader_kwargs)}#, "celeba": DataLoader(train_dataset2, shuffle=True, **self.loader_kwargs)}

    def val_dataloader(self) -> DataLoader:
        # Create val dataset and return loader
        val_dataset = NafnetDataset(
            target="val",
        )

        val_dataset2 = NafnetGANDataset(target="val")

        return [DataLoader(val_dataset, shuffle=True, **self.loader_kwargs)]#, DataLoader(val_dataset2, shuffle=True, **self.loader_kwargs)]

    def test_dataloader(self) -> DataLoader:
        # Create test dataset and return loader
        test_dataset = NafnetDataset(
            target="test",
        )

        test_dataset2 = NafnetGANDataset(
            target="test",
        )
        return [DataLoader(test_dataset, **self.loader_kwargs)]#, DataLoader(test_dataset2, **self.loader_kwargs)]