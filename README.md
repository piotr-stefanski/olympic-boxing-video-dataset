# olympic-boxing-video-dataset

# Pre requirements
1. Install `uv` package manager: https://docs.astral.sh/uv/getting-started/installation/
2. Install dependencies, run: `uv sync`
3. Download database from repository: 
   1. (Temporary link, TO REPLACE) https://drive.google.com/drive/folders/1s7KxrnJg_1CigQVIIghQ2T3Hty5NIufm?usp=sharing
   2. For this you can use `gdown` package, run following command in terminal:
   ```bash
    uv run gdown --folder 1s7KxrnJg_1CigQVIIghQ2T3Hty5NIufm -O ../database
    ```
4. Adjust $DATABASE_PATH environment variable if you put repository in another location than `../database`

# Initialization
1. Convert original database to coco format by running script:
```bash
uv run convert_database_to_coco_format.py --save-images
```
⚠️ Be carefully `--save-images` flag extract frames from video and save it in original size to the disk, you need **~156GB** of free disk to that operation.

⚠️ This command is not well optimized and takes some time.

# Quick Start - Testing

> [!IMPORTANT]
> **GPU cluster connection is almost necessary** for testing, as the data loading uses NVIDIA DALI which requires GPU access.

## 1. Connect to GPU node
```bash
srun -p plgrid-gpu-a100 -N 1 -c 3 --gpus=1 --pty /bin/bash -l
```

## 2. Test dataset (DALI loader)
```bash
uv run python dataset.py
```
This verifies DALI can load images and annotations correctly.

## 3. Test model
```bash
uv run python model.py
```
This initializes Faster R-CNN and prints model summary.

## 4. Test training (dry run)
```bash
uv run python train.py --dry_run --print_model
```
This runs one iteration to verify the full pipeline works.

# The main objective of the projekt
Project was created to prepare dataset for training object detection models and also prepare code to train such models. There are plans to train several models and compare their performance. In the beginning we will start with the state of the art models like R-CNN, YOLO etc.

## Available resources to train models
To prepare benchamrk we have access to the ACK Cyfronet datacenter, especially to the Athena cluster (https://docs.cyfronet.pl/spaces/~plgpawlik/pages/126648338/Athena). Basically we start running training only on one node which offer access to the up to 8 Nvidia A100 GPUs.

So the code being prepared need to be well optimized in every stage. We have access to the fast GPU, so reading and encoding images from disk and load it to the GPU need be fast. As we already tested the "standard" Pytorch DataLoader is not enought and cannot keep up with delivering data to the GPU, resulting in the graphics card being used at 50/60% capacity.

## Plan of implementation
1. Prepare the code to review samples from the dataset.
2. Prepare the data loader which is highly efficient and can keep up with delivering data to the GPU. Database has images in full hd resolution, so encoding it only by the cpu is not enough. Try to load data using multiple threads and encode it using GPU. Also take into account images is split into 5 folds in annotation level (look at convert_database_to_coco_format.py for more details).
So in the implementation prepare loader to get number of folds as parameter and return only images from selected fold. In default implementation first 4 folds is used for training and last fold is used for validation.
3. Prepare the R-CNN model to train on the dataset and display their structure, layers, number of parameters etc.
4. Prepare the training pipeline to train the model on the dataset.
5. Prepare the evaluation pipeline to evaluate the model on the dataset.


Firstly we want to implement the training pipeline on our own to have full control about such process. The library under consideration for this is PyTorch.

After that we consider using MMdetection (https://github.com/open-mmlab/mmdetection) to process whole benchmark.