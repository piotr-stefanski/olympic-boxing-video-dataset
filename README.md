# olympic-boxing-video-dataset

# Pre requirements
1. Install `uv` package manager: https://docs.astral.sh/uv/getting-started/installation/
2. Install dependencies, run: `uv sync`
3. Download database from repository: (Temporary link, TO REPLACE) https://drive.google.com/drive/folders/1s7KxrnJg_1CigQVIIghQ2T3Hty5NIufm?usp=sharing
4. Adjust $DATABASE_PATH environment variable if you put repository in another location than `../database`

# Initialization
1. Convert original database to coco format by running script:
```bash
uv run convert_database_to_coco_format.py --save-images
```
Be carefully `--save-images` flag extract frames from video and save it in original size to the disk, you need ~110GB of free disk to that operation.