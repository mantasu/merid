# MERID: Mask-based Eyeglasses Removal via Inpainting and Denoising

## About

The source code for the MLP Coursework 4. Our proposed model MERID is able to remove glasses and sunglasses from unconstrained environments and various angles. It has show to achieve state-of-the-art results in terms of realism and identity preservation measures. This package contains files that define our MERID model pipeline as well as provides scripts to set up the data and train the model.

> **Note**: the best model is currently being trained (the number of parameters is increased). The total training time is estimated to be 5 days on a single GTX 3080 Ti. 

## Quick Start

To run a quick inference to see how the model works, setup the environment and download all the model weights as described below, then run the following (you can put your own $256 \times 256$ images inside [demo](data/demo) directory):

```shell
python scripts/predict.py -c config.json -i data/demo
```

## Setup

The code was built and tested using [Python 3.10.9](https://www.python.org/downloads/release/python-3109/) It is recommended to setup [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:
```bash
conda create -n merid python=3.10
conda activate merid
```

The environment uses [Pytorch 1.13](https://pytorch.org/blog/PyTorch-1.13-release/) with [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive). Please, also install the required packages (may take some time):
```bash
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Models

To keep the credid where credid is due, we require to download weights for certain parts of the architecture from the original author's host link. Otherwise, we share our weights via Google Drive. Please put all the downloaded files inside `checkpoints` directory:

1. [`pretrained.pt`](https://drive.google.com/file/d/1Ea8Swdajz2J5VOkaXIw_-pVJk9EWYrpx/view) - Portrait Eyeglasses Removal weights for mask generation. Original repository is [here](https://github.com/StoryMY/take-off-eyeglasses).
2. [`celeba_hq.ckpt`](https://drive.google.com/drive/folders/1cSCTaBtnL7OIKXT4SVME88Vtk4uDd_u4) - DDNM weights on CelebA for inpainting. Original repository is [here](https://github.com/wyhuai/DDNM).
3. [`InpaintingModel_gen.pth` and `landmark_detector.pth`](https://drive.google.com/drive/folders/1Xwljrct3k75_ModHCkwcNjJk3Fsvv-ra) - LaFIn weights on CelebA-HQ for inpainting. Original repository is [here](https://github.com/YaN9-Y/lafin).
4. All 4 `.pth` models from the following [link](https://drive.google.com/file/d/1U9OCL6g6H1LAQx41obGsSXoSlCE2K_in/view?usp=sharing). We provide our trained weights, however, the best model is still being finetuned.

## Datasets

The instructions are provided for Linux users. Please enable executive privilages for python scripts. Also, you may want to install the unzipping packages, e.g., for Ubuntu:
```bash
chmod +x ./scripts/*.py
sudo apt-get install p7zip-full unzip
```

Once all the datasets are downloaded and preprocessed, the data structure should loke as follows:
```
data                             <- The data directory under project
│   ├── celeba
│   │   └── test
│   │   |   └── no_glasses       <- 256x256 images of poeple without sunglasses
│   │   |   └── glasses          <- 256x256 images of poeple with sunglasses
│   │   |   └── masks            <- 256x256 images of glasses masks
│   │   |
|   |   └── train
│   │   |   └── no_glasses       <- 256x256 images of poeple without sunglasses
│   │   |   └── glasses          <- 256x256 images of poeple with sunglasses
│   │   |   └── masks            <- 256x256 images of glasses masks
│   │   |
|   |   └── val
│   │       └── no_glasses       <- 256x256 images of poeple without sunglasses
│   │       └── glasses          <- 256x256 images of poeple with sunglasses
│   │       └── masks            <- 256x256 images of glasses masks
│   │
│   ├── celeba-mask-hq
│   │   └── test                 <- Same as celeba except without no_glasses
|   |   └── train                <- Same as celeba except without no_glasses
|   |   └── val                  <- Same as celeba except without no_glasses
|   |
│   ├── meglass
│   │   └── test                 <- Same as celeba except without masks
|   |
│   ├── lfw
│   │   └── test                 <- Same as celeba except without masks
|   |
│   ├── synthetic
│   │   └── test                 <- Same as celeba
|   |   └── train                <- Same as celeba
|   |   └── val                  <- Same as celeba

```

<details><summary><h3>Preparing Training Datsets</h3></summary>

#### CelebA Mask HQ (optional)

1. Download the files from Google Drive:
    * Download `CelebAMask-HQ.zip` folder from [here](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) and put it under `data/celeba-mask-hq/CelebAMask-HQ.zip`
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

#### Synthetic

1. Download the files from Google Drive:
    * Download `ALIGN_RESULT_V2.zip` from [here](https://drive.google.com/file/d/1X1qkozQbVyz5lUA8xd-lYfy1jauOji46/view) and place it under `data/synthetic/ALIGN_RESULT_V2.zip`
    * Download `synthetic_augment.zip` from [here](https://drive.google.com/file/d/1wqpiSaoiuWEm8fi2xKne40jtdpQlItGR/view?usp=sharing) and place it under `data/synthetic/synthetic_augment.zip`
2. Unzip the data
    ```bash
    unzip data/synthetic/ALIGN_RESULT_v2.zip -d data/synthetic
    unzip data/synthetic/synthetic_augment.zip -d data/synthetic
    ```
3. Generate shadow labels and split to glasses and their labels:
    ```bash
    python scripts/preprocess_synthetic.py
    ```
4. Cleanup the workspace:
    ```bash
    rm -rf data/synthetic/ALIGN_RESULT_v2 data/synthetic/ALIGN_RESULT_v2.zip data/synthetic/synthetic_augment.zip
    ```

#### CelebA

1. Download the files from Google Drive:
    * Download `img_celeba.7z` folder from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28?resourcekey=0-f5cwz-nTIQC3KsBn3wFn7A) and put it under `data/celeba/img_celeba.7z`
    * Download `annotations.zip` file from [here](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view) and put it under `data/celeba/annotations.zip` (_Note:_ keep `standard_landmark_68pts.txt` for **LFW** and **MeGlass** datasets)
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
    rm -rf data/celeba/img_celeba.7z data/celeba/img_celeba
    rm data/celeba/annotations.zip data/celeba/*.txt
    ```

</details>


<details><summary><h3>Preparing Test Datsets</h3></summary>

### FFHQ

1. Download the resized data from Kaggle, face model from GitHub and its weights form Google Drive:
    * Download `archive.zip` from [here](https://www.kaggle.com/datasets/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px) and put it under `data/ffhq/archive.zip`
    * Download `face-parsing.PyTorch-master.zip` from [here](https://github.com/zllrunning/face-parsing.PyTorch) and place it under `data/ffqh/face-parsing.PyTorch-master.zip`
    * Download `79999_iter.pth` from [here](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) and put it under `data/ffhq/79999_iter.pth`
2. Unzip the data:
    ```bash
    unzip data/ffhq/archive.zip -d data/ffhq
    unzip data/ffhq/face-parsing.PyTorch-master.zip -d data/ffhq
    ```
3. Crop, align and split to glasses/no-glasses:
    ```bash
    python scripts/preprocess_ffhq.py
    ```
4. Clean up
    ```bash
    rm -rf data/ffhq/resized data/ffhq/face-parsing.PyTorch-master
    rm data/ffhq/archive.zip data/ffhq/face-parsing.PyTorch-master.zip data/ffhq/79999_iter.pth
    ```

### LFW

1. Download the files from the official host:
    * Download `lfw.tgz` from [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz) and put it under `data/lfw/lfw.tgz`
    * Download `lfw_attributes.txt` from [here](https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt) and place it under `data/lfw/lfw_attributes.txt`
    * Download the 68 landmarks predictor from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it under `data/lfw/shape_predictor_68_face_landmarks.dat.bz2` (keep it for **MeGlass**)
    * Download `standard_landmark_68pts.txt` from [here](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view) and place under `data/celeba/standard_landmark_68pts.txt` (yes, under `celeba`, not `lfw` - you may already have it)
2. Unzip the data:
    ```bash
    tar zxvf ./data/lfw/lfw.tgz -C data/lfw
    bunzip2 data/lfw/shape_predictor_68_face_landmarks.dat.bz2
    ```
3. Split the dataset:
    ```bash
    python scripts/preprocess_lfw.py
    ```
4. Clean up the directory
    ```bash
    rm -rf ./data/lfw/lfw
    rm data/lfw/lfw.tgz data/lfw/lfw_attributes.txt data/lfw/shape_predictor_68_face_landmarks.dat
    ```

#### MeGlass

1. Download the files from Baidu Yun and Github:
    * Download `MeGlass_ori.zip` from [here](https://pan.baidu.com/s/17EBZz3LkQzyn44VL45udTg) and place it under `data/meglass/MeGlass_ori.zip`
    * Download all `.txt` files from [here](https://github.com/cleardusk/MeGlass/tree/master/test) and place them under `data/meglass/*.txt`
    * Download the 68 landmarks predictor from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it under `data/lfw/shape_predictor_68_face_landmarks.dat.bz2` (yes, under `lfw`, not `meglass` - you may already have it)
    * Download `standard_landmark_68pts.txt` from [here](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view) and place under `data/celeba/standard_landmark_68pts.txt` (yes, under `celeba`, not `meglass` - you may already have it)
2. Unzip the data
    ```bash
    unzip data/meglass/MeGlass_ori.zip -d data/meglass
    bunzip2 data/lfw/shape_predictor_68_face_landmarks.dat.bz2
    ```
3. Crop, align and split to glasses/no-glasses:
    ```bash
    python scripts/preprocess_meglass.py
    ```
4. Clean up the directory:
    ```bash
    rm -rf data/meglass/MeGlass_ori data/meglass/*.txt
    rm data/meglass/MeGlass_ori.zip data/meglass/shape_predictor_68_face_landmarks.dat
    ```

</details>


## Training

Training is currently supported for `3` sub-architectures - [sunglasses segmenter](src/models/merid/sunglasses_segmenter.py), [denoiser](src/models/nafnet/nafnet_denoiser.py) and [recolorizer](src/models/merid/recolorizer.py). To train the sunglasses classifier, please check out a neighboring repository [sunglasses-or-not](https://github.com/mantasu/sunglasses-or-not). To perform training, just call the corresponding files (you can adjust the parameters like batch size, learning rate inside files directly).

## Evaluation

To evaluate the whole MERID model, please first generate the results without glasses. For example, evaluating on FFHQ:
```
python -m pytorch_fid data/ffhq/test/no_glasses data/ffhq/test/no_glasses_generated --device cuda:0
```

> To evaluate the model on identity preservation, please refer to [this file](scripts/FaceReconRank1Acc.py).

For evaluating specific sub-architectures on their trained datasets (specifically, test splits), please also refer to the specific model files.

## Config

The config file is simply a dictionary with 3 entries for the three stages as explained in the paper: mask generation, mask enhancement and masked inpainting. An example setup with all pretrained models can be seen in [config.json](config.json). Each part contains 2 dictionaries: `modules` and `wights`:

* `modules`: a dictionary of modules for each pipeline part. The specific names are listed in [config.json](config.json). Each module is another dictionary containing `name` entry specifying the models class that should be loaded and any extra parameters that go inside that class.
* `weights`: a dictionary with corresponding weight entries for the models specified in `modules`. Each entry is a dictionary that contains 3 possible items: `path` - the path to actual weights (can be a list if weights are nested as dictionary entires), `freeze` - whether to freeze the loaded model with weights, and `guest_fn` - the name of the function to apply to weights to preprocess them (see [config.py](src/utils/config.py) for more details).

> **Note**: You may want to replace the DDNM inpainter provided in the default `config.json` with LaFIn inpainter for a much faster inference. If so, replace the `inpainter` parts inside `modules` and `weights` with th efollowing lines:

```json
"modules": {
    "inpainter": {"name": "LafinInpainter", "det_weights": "checkpoints/landmark_detector.pth", "gen_weights": "checkpoints/InpaintingModel_gen.pth"}
},
"weights": {
    "inpainter": {"path": null, "freeze": true}
}
```

## Reference Repositories

The work was heavily influenced and a lot of code has been borrowed and modified from the following repositories:
* **[HD-CelebA-Cropper](https://github.com/LynnHo/HD-CelebA-Cropper)** - for cropping and aligning CelebA face images
* **[take-off-eyeglasses](https://github.com/StoryMY/take-off-eyeglasses)** - for serving as the first part of our architecture, i.e., mask generation
* **[DDNM](https://github.com/wyhuai/DDNM)** - for diffusive inainting part of our architecture
* **[lafin](https://github.com/YaN9-Y/lafin)** - for generative inpainting part of our architecture
* **[NAFNet](https://github.com/megvii-research/NAFNet)** - for denoising part of our architecture
