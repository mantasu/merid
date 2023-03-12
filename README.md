# RemGlass: Getting Rid of Eyeglasses Through Pretrained Diffusion

## About

* [ ] Write description
* [ ] Insert banner

## Setup

The code was built and tested using [Python 3.10.9](https://www.python.org/downloads/release/python-3109/) It is recommended to setup [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:
```bash
conda create -n remglass python=3.10
conda activate remglass
```

The environment uses [Pytorch 1.13](https://pytorch.org/blog/PyTorch-1.13-release/) with [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive). Please, also install the required packages (may take some time):
```bash
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Datasets

The instructions are provided for Linux users. Please enable executive privilages for python scripts. Also, you may want to install the unzipping packages, e.g., for Ubuntu:
```bash
chmod +x ./scripts/*.py
sudo apt-get install p7zip-full unzip
```

Once all the datasets are downloaded and preprocessed, the data structure should loke as follows:
```
├── data                <- The data directory under root
│   ├── celeba
│   │   └── train_x     <- 256x256 images with glasses
|   |   └── train_y     <- 256x256 images without glasses
│   │
│   ├── lfw
│   │   └── test_x      <- 256x256 images with glasses
|   |   └── test_y      <- 256x256 images without glasses
│   │
│   ├── meglass
│   │   └── test_x      <- 256x256 images with glasses
|   |   └── test_y      <- 256x256 images without glasses
│   │
│   ├── synthetic
│   │   └── train_x     <- 256x256 images with glasses
|   |   └── train_y     <- 256x256 masks and images without glasses

```

<details><summary><h3>Preparing CelebA Mask HQ (optional)</h3></summary>

1. Download the files from Google Drive:
    * Download `CelebAMask-HQ.zip` folder from [here](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) and put it under `./data/celeba-mask-hq/CelebAMask-HQ.zip`
    * Download `annotations.zip` file from [here](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view) and put it under `data/celeba/annotations.zip` (_Note:_ you will need this file for `celeba`, so just put it there, _not_ in `celeba-mask-hq`)
3. Unzip the data:
    ```bash
    unzip data/celeba-mask-hq/CelebAMask-HQ.zip -d data/celeba-mask-hq
    ```
4. Split to train/val/test
    ```bash
    python scripts/preprocess_celeba_mask_hq.py
    ```
5. Clean up
    ```bash
    rm -rf data/celeba-mask-hq/CelebAMask-HQ data/celeba-mask-hq/CelebAMask-HQ.zip
    ```

</details>

<details><summary><h3>Prepareing Synthetic Dataset</h3></summary>

1. Download the files from Google Drive:
    * Download `ALIGN_RESULT_V2.zip` from [here](https://drive.google.com/file/d/1X1qkozQbVyz5lUA8xd-lYfy1jauOji46/view) and place it under `data/synthetic/ALIGN_RESULT_V2.zip`
2. Unzip the data
    ```bash
    unzip data/synthetic/ALIGN_RESULT_v2.zip -d data/synthetic
    ```
3. Generate shadow labels and split to glasses and their labels:
    ```bash
    python scripts/preprocess_synthetic.py
    ```
4. Cleanup the workspace:
    ```bash
    rm -rf ./data/synthetic/ALIGN_RESULT_v2 data/synthetic/ALIGN_RESULT_v2.zip
    ```

</details>

<details><summary><h3>Preparing CelebA</h3></summary>

1. Download the files from Google Drive:
    * Download `img_celeba.7z` folder from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28?resourcekey=0-f5cwz-nTIQC3KsBn3wFn7A) and put it under `./data/celeba/img_celeba.7z`
    * Download `annotations.zip` file from [here](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view) and put it under `./data/celeba/annotations.zip`
3. Unzip the data:
    ```bash
    7z x data/celeba/img_celeba.7z/img_celeba.7z.001 -o./data/celeba
    unzip data/celeba/annotations.zip -d data/celeba/
    ```
4. Crop, align and split to glasses/no-glasses:
    ```bash
    python scripts/preprocess_celeba.py
    ```
5. Clean up
    ```bash
    rm -rf ./data/celeba/img_celeba.7z ./data/celeba/img_celeba ./data/celeba/aligned
    rm ./data/celeba/annotations.zip ./data/celeba/*.txt
    ```

</details>

<details><summary><h3>Preparing MeGlass</h3></summary>

1. Download the files from Baidu Yun and Github:
    * Download `MeGlass_ori.zip` from [here](https://pan.baidu.com/s/17EBZz3LkQzyn44VL45udTg) and place it under `./data/meglass/MeGlass_ori.zip`
    * Download all `meta.txt` from [here](https://github.com/cleardusk/MeGlass) and place it under `./data/meglass/meta.txt`
2. Unzip the data
    ```bash
    unzip ./data/meglass/MeGlass_ori.zip -d ./data/meglass/
    ```
3. Split the dataset:
    ```bash
    python ./scripts/split.py --dataset meglass --resize_h 256 --resize_w 256
    ```
4. Clean up the directory:
    ```bash
    rm -rf ./data/meglass/MeGlass_ori
    rm ./data/meglass/MeGlass_ori.zip ./data/meglass/meta.txt
    ```

</details>

<details><summary><h3>Preparing LFW</h3></summary>

1. Download the files from the official host:
    * Download `lfw-deepfunneled.tgz` from [here](http://vis-www.cs.umass.edu/lfw/#deepfunnel-anchor) and place it under `./data/lfw/lfw-deepfunneled.tgz`
    * Download `lfw_attributes.txt` from [here](https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt) and place it under `./data/lfw/lfw_attributes.txt`
2. Unzip the data:
    ```bash
    tar zxvf ./data/lfw/lfw-deepfunneled.tgz -C ./data/lfw/
    ```
3. Split the dataset:
    ```bash
    python ./scripts/split.py --dataset lfw --resize_h 256 --resize_w 256
    ```
4. Clean up the directory
    ```bash
    rm -rf ./data/lfw/lfw-deepfunneled
    rm ./data/lfw/lfw-deepfunneled.tgz ./data/lfw/lfw_attributes.txt
    ```

</details>

## Checkpoints

For training, certain pre-trained models are used to initialize parts of our architecture. Please download and put the files under `checkpoints` directory:
1. `vgg_normalised.pth` - download from [here](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view). Used to initialize 

## File Structure
* [ ] Describe file structure

## Training
* [ ] Describe how to train the model

## Testing
* [ ] Describe how to evaluate the model



<details><summary><h3>FID</h3></summary>




1. Install Package:

   ```bash
   pip install pytorch-fid
   ```

2. Run:

   ```bash
   python -m pytorch_fid data/meglass/test_x data/lfw/test_x --device cuda:0
   ```

   ```bash
   (base) ➜  remglass git:(main) ✗ python -m pytorch_fid data/meglass/test_x data/lfw/test_x --device cuda:0
   100%|█████████████████████████████████████████████████████████████████████████████████████████████| 297/297 [00:16<00:00, 18.27it/s]
   100%|█████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  8.40it/s]
   FID:  180.22132973432053
   ```

</details>


</details>

## Demo
To run a demo, put an image to `demo/` and modify [inference.py](inference.py) constants at the top to match the desired behavior. Run the following to generate an image with the removed object:
```bash
python inference.py
```

## Config

### Domain Adapter
* `is_torchvision_vgg`: whether to load [torchvision _VGG-19_](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html#torchvision.models.vgg19) model or to use the custom [normalised _VGG-19_](https://github.com/naoto0804/pytorch-AdaIN) as in the original "Portrait Eyeglasses Removal" paper
* `vgg_weights`: the path to VGG weights. Please specify one of the following 3 options:
    * Leave empty `""` to keep random initialization, e.g., if weights for Domain Adapter are going to be loaded (would automatically initialize VGG weights).
    * If `is_torchvision_vgg` is `false`, set to the path of the downloaded weights from the repository which contains that _VGG-19_ version
    * If `is_torchvision_vgg` is `true`, set to one of the [torchvision's](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html#torchvision.models.VGG19_Weights) options like `"DEFAULT"`
* `weights`: the path to domain adapter weights. If the parameters file contains weights for multiple modules, instead of a path, specify as a list, where the first entry is the path and the second entry is the dictionary key for domain adapter weights. If the value is empty `""`, then it will be trained.

## References
The work was heavily influenced and a lot of code has been borrowed and modified from the following repositories:
* **[HD-CelebA-Cropper](https://github.com/LynnHo/HD-CelebA-Cropper)** - for cropping and aligning CelebA face images