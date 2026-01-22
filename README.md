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

# The main objective of the projekt
Project was created to prepare dataset for training object detection models and also prepare code to train such models. There are plans to train several models and compare their performance. In the beginning we will start with the state of the art models like R-CNN, YOLO etc.

## Available resources to train models
To prepare benchamrk we have access to the ACK Cyfronet datacenter, especially to the Athena cluster (https://docs.cyfronet.pl/spaces/~plgpawlik/pages/126648338/Athena). Basically we start running training only on one node which offer access to the up to 8 Nvidia A100 GPUs.

So the code being prepared need to be well optimized in every stage. We have access to the fast GPU, so reading and encoding images from disk and load it to the GPU need be fast. As we already tested the "standard" Pytorch DataLoader is not enought and cannot keep up with delivering data to the GPU, resulting in the graphics card being used at 50/60% capacity.