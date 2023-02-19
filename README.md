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

<details><summary><h3>Prepareing Synthetic Dataset</h3></summary>

1. Download the files from Google Drive:
    * Download `ALIGN_RESULT_V2.zip` from [here](https://drive.google.com/file/d/1X1qkozQbVyz5lUA8xd-lYfy1jauOji46/view) and place it under `./data/synthetic/ALIGN_RESULT_V2.zip`
    * Download `basic_split.txt` from [here](https://drive.google.com/file/d/1ahqlo03laA3edlH0jMgcgIpHki4WiNaH/view) and place it under `./data/synthetic/basic_split.txt`
2. Unzip the data
    ```bash
    unzip ./data/synthetic/ALIGN_RESULT_v2.zip -d ./data/synthetic
    ```
3. Generate shadow labels and split to glasses and their labels:
    ```bash
    python ./scripts/gen_shadows.py --syndata_dir ./data/synthetic/ALIGN_RESULT_v2
    python ./scripts/split.py --dataset synthetic
    ```
4. Cleanup the workspace:
    ```bash
    rm -rf ./data/synthetic/ALIGN_RESULT_v2
    rm ./data/synthetic/ALIGN_RESULT_v2.zip ./data/synthetic/basic_split.txt
    ```

</details>

<details><summary><h3>Preparing CelebA</h3></summary>

1. Download the files from Google Drive:
    * Download `img_celeba.7z` folder from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28?resourcekey=0-f5cwz-nTIQC3KsBn3wFn7A) and put it under `./data/celeba/img_celeba.7z`
    * Download `annotations.zip` file from [here](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view) and put it under `./data/celeba/annotations.zip`
    * Download `list_attr_celeba.txt` file from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ) and put it under `./data/celeba/list_attr_celeba.txt`
3. Unzip the data:
    ```bash
    7z x ./data/celeba/img_celeba.7z/img_celeba.7z.001 -o./data/celeba/
    unzip ./data/celeba/annotations.zip -d ./data/celeba/
    ```
4. Crop, align and split to glasses/no-glasses:
    ```bash
    python ./scripts/align_celeba.py --crop_size_h 256 --crop_size_w 256 --order 4 --n_worker 24
    python ./scripts/split.py --dataset celeba
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

## File Structure
* [ ] Describe file structure

## Training
* [ ] Describe how to train the model

## Testing
* [ ] Describe how to evaluate the model

## Demo
To run a demo, put an image to `demo/` and modify [inference.py](inference.py) constants at the top to match the desired behavior. Run the following to generate an image with the removed object:
```bash
python inference.py
```

## References
The work was heavily influenced and a lot of code has been borrowed and modified from the following repositories:
* **[HD-CelebA-Cropper](https://github.com/LynnHo/HD-CelebA-Cropper)** - for cropping and aligning CelebA face images