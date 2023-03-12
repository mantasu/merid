import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from utils.io import load_json
from train import parse_arguments, prepare_model, prepare_datamodule

from architectures.mask_inpainter import MaskInpainter

def overlay(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    # Use the correct types
    mask = mask.astype(np.uint8)
    
    # Create the overlay colour
    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    # Init overlayed image
    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Create a completely overlayed image
        overlay = np.ones(image.shape) * np.array(colors[object_id])
        foreground = alpha * image + (1 - alpha) * overlay
        
        # Only keep the overlay on masked part
        binary_mask = mask == object_id
        im_overlay[binary_mask] = foreground[binary_mask]

        # Emphasize the contours in the segmented regions
        countours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours, :] = 0
    
    # Compose an overlayed image
    overlayed = im_overlay.astype(image.dtype)

    return overlayed

def tensor_to_image(tensor):
    if (tensor < 0).any():
        tensor = (tensor + 1.0) / 2.0
    
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    
    if tensor.shape[0] == 3:
        tensor = np.moveaxis(tensor, 0, -1)
    
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze()

    return Image.fromarray(tensor)

from skimage.morphology import disk, binary_dilation, binary_erosion, star, binary_closing, area_closing
from skimage import morphology


def dilate(binary_mask, max_strength=6, num_iter=2):
    strength = max_strength
    
    for _ in range(num_iter):
        if strength < 2:
            break

        binary_mask = binary_dilation(binary_mask, disk(strength))
        binary_mask = binary_erosion(binary_mask, disk(strength // 3))
        strength -= 2

        for _ in range(strength // 2):
            binary_mask = binary_erosion(binary_mask, star(1))
            binary_mask = binary_erosion(binary_mask, disk(1))
    
    return binary_mask


if __name__ == "__main__":
    config = load_json(parse_arguments().config)

    model = prepare_model("mask_generator", config["mask_generator"])
    datamodule = prepare_datamodule(config["data"])
    dataloader = datamodule.val_dataloader()

    nnn = 8
    it = iter(dataloader)
    # next(it)

    x1, x2, y1, y2 = next(it)
    sample = x1

    from extra.sunglasses_classifier import SunglassesClssifier
    sun = SunglassesClssifier("checkpoints/sunglasses-classifier-best.pth")
    preds = sun(sample).sigmoid().round().squeeze()
    print(preds)
    


    out_glasses, out_shadows = model(sample)
    x2, out_glasses, out_shadows = sample[:nnn], out_glasses[:nnn], out_shadows[:nnn]
    mask_glasses = out_glasses.argmax(1).numpy().astype(np.uint8)
    mask_shadows = out_shadows.argmax(1).numpy().astype(np.uint8)

    smooth_glasses = mask_glasses.copy()
    smooth_shadows = mask_shadows.copy()

    from extra.sunglasses_classifier import MaskPostprocesser
    pper = MaskPostprocesser("checkpoints/sunglasses-classifier-best.pth")
    print(out_glasses.argmax(1).shape)
    pper_mask = pper(x2, out_glasses.argmax(1).unsqueeze(1), out_shadows.argmax(1).unsqueeze(1))


    for l, (is_sun, binary_glasses, binary_shadows) in enumerate(zip(preds, mask_glasses, mask_shadows)):
        
        # binary_glasses = binary_erosion(binary_glasses, disk(1))

        # binary_shadows = binary_dilation(binary_shadows, disk(7))
        # binary_shadows = binary_erosion(binary_shadows, star(2))
        # binary_shadows = binary_erosion(binary_shadows, disk(1))
        # binary_shadows = binary_erosion(binary_shadows, star(1))
        # binary_shadows = binary_erosion(binary_shadows, disk(1))
        # binary_shadows = binary_dilation(binary_shadows, star(5))
        # binary_shadows = binary_erosion(binary_shadows, disk(2))
        # binary_shadows = binary_erosion(binary_shadows, star(2))
        # binary_shadows = binary_erosion(binary_shadows, disk(1))
        # binary_shadows = binary_erosion(binary_shadows, star(1))
        # binary_shadows = binary_erosion(binary_shadows, disk(1))

        # for i in range(3):
        #     binary_image = binary_erosion(binary_image, diamond(1))

        if is_sun == 1:
            # smooth_glasses[l] = binary_dilation(smooth_glasses[l], star(5))
            smooth_glasses[l] = binary_dilation(smooth_glasses[l], disk(5))
            smooth_glasses[l] = morphology.binary_closing(smooth_glasses[l], disk(20))
            # smooth_glasses[l] = morphology.remove_small_holes(smooth_glasses[l], 256)
        else:
            smooth_glasses[l] = binary_dilation(smooth_glasses[l], disk(1)) # dilate(binary_glasses)
            smooth_shadows[l] = binary_dilation(smooth_shadows[l], disk(1)) # dilate(binary_shadows)
    

    imgs = x2
    smooth_both = (smooth_glasses | smooth_shadows)
    mask_both = (mask_glasses | mask_shadows)

    smooth_both = pper_mask.squeeze().numpy().astype(np.uint8)
    print(smooth_both.shape)

    

    from architectures.ddnm.ddnm_inpainter import DDNMInpainter
    diff = DDNMInpainter().to("cuda:0")
    diff.load_state_dict(torch.load("checkpoints/celeba_hq.ckpt"))
    mask_inpainter = MaskInpainter(diff, config={"core_inpainter": diff})

    print("Ready to inpaint")

    from architectures.lafin.lafin_inpainter import LafinInpainter

    inp = None

    lafin = LafinInpainter("checkpoints/landmark_detector.pth", "checkpoints/InpaintingModel_gen.pth").cuda()
    inp = lafin(imgs.cuda(), (torch.tensor(smooth_both).unsqueeze(1) > 0).float().cuda()).cpu()
    

    # inp = mask_inpainter(imgs.cuda(), torch.tensor(1 - smooth_both).unsqueeze(1).cuda()).cpu()
    # inp_bad = mask_inpainter(imgs.cuda(), torch.tensor(1 - mask_both).unsqueeze(1).cuda())
    
    # from diffusers import StableDiffusionInpaintPipeline
    # from PIL import Image
    # from tqdm import tqdm
    # from utils.io_and_types import tensor_to_image as tti

    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-inpainting",
    #     revision="fp16",
    #     torch_dtype=torch.float16,
    # )
    # pipe
    # prompt = "Eye region, no glasses"
    # #image and mask_image should be PIL images.
    # #The mask structure is white for inpainting and black for keeping as is

    # inps = []
    # for img, msk in tqdm(zip(imgs, smooth_both)):
    #     im = pipe(prompt=prompt, image=tti(img, as_pil=True), mask_image=tti(torch.tensor(msk), as_pil=True)).images[0]
    #     inps.append(np.array(im))
    
    # inp = inps
    from torchvision.transforms.functional import to_pil_image



    plt.figure(figsize=(20, 20))

    for i, img in enumerate(imgs):
        img = tensor_to_image(img)
        msk_glasses = Image.fromarray(mask_glasses[i] * 255)
        smh_glasses = Image.fromarray(smooth_glasses[i] * 255)
        msk_shadows = Image.fromarray(mask_shadows[i] * 255)
        smh_shadows = Image.fromarray(smooth_shadows[i] * 255)
        smh_both = Image.fromarray(smooth_both[i] * 255)
        msk = Image.fromarray(overlay(np.array(img), smooth_both[i]))
        inpainted = to_pil_image(inp[i]) # tensor_to_image(inp[i])
        # ino = tensor_to_image(inp_bad[i])
        #  truth = tensor_to_image(y3[i])

        gla_tru = tensor_to_image(y1[i]) # Image.fromarray(y1[i].numpy().astype(np.uint8) * 255)
        sha_tru = tensor_to_image(y2[i]) # Image.fromarray(y2[i].numpy().astype(np.uint8) * 255)

         # truth.save(f"ddnm_examples/celeba/{i}-ground-truth-inpaint.jpg", "JPEG")
        # gla_tru.convert("L").save(f"ddnm_examples/celeba/{i}-ground-truth-glasses.jpg", "JPEG")
        # sha_tru.convert("L").save(f"ddnm_examples/celeba/{i}-ground-truth-shadows.jpg", "JPEG")
        # Image.fromarray((y1[i].numpy().astype(np.uint8) | y2[i].numpy().astype(np.uint8)) * 255).convert("L").save(f"ddnm_examples/celeba/{i}-ground-truth-full-mask.jpg", "JPEG")

        # img.save(f"ddnm_examples/celeba/{i}-input.jpg", "JPEG")
        # msk_glasses.convert("L").save(f"ddnm_examples/celeba/{i}-output-glasses.jpg", "JPEG")
        # smh_glasses.convert("L").save(f"ddnm_examples/celeba/{i}-output-glasses-smooth.jpg", "JPEG")
        # msk_shadows.convert("L").save(f"ddnm_examples/celeba/{i}-output-shadows.jpg", "JPEG")
        # smh_shadows.convert("L").save(f"ddnm_examples/celeba/{i}-output-shadows-smooth.jpg", "JPEG")
        # Image.fromarray((mask_glasses[i] | mask_shadows[i]) * 255).convert("L").save(f"ddnm_examples/celeba/{i}-output-full-mask.jpg", "JPEG")
        # smh_both.convert("L").save(f"ddnm_examples/celeba/{i}-output-full-mask-smooth.jpg", "JPEG")
        # inpainted.save(f"ddnm_examples/celeba/{i}-output-inpainted-smooth.jpg", "JPEG")
        # ino.save(f"ddnm_examples/celeba/{i}-output-inpainted-not-smooth.jpg", "JPEG")


        # plt.subplot(4, len(imgs), 0 * len(imgs) + i + 1)
        # plt.imshow(img)
        plt.subplot(4, len(imgs), 0 * len(imgs) + i + 1)
        plt.imshow(msk_glasses)
        # plt.subplot(8, len(imgs), 2 * len(imgs) + i + 1)
        # plt.imshow(smh_glasses)
        # plt.subplot(8, len(imgs), 3 * len(imgs) + i + 1)
        # plt.imshow(msk_shadows)
        # plt.subplot(8, len(imgs), 4 * len(imgs) + i + 1)
        # plt.imshow(smh_shadows)
        plt.subplot(4, len(imgs), 1 * len(imgs) + i + 1)
        plt.imshow(smh_both)
        plt.subplot(4, len(imgs), 2 * len(imgs) + i + 1)
        plt.imshow(msk)
        plt.subplot(4, len(imgs), 3 * len(imgs) + i + 1)
        plt.imshow(inpainted)

    plt.show()
