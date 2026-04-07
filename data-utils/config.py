import os

EXPECTED_TOTAL_NUMBER_OF_FRAMES = 630965
FOLDS_DEFINITION = {
    'fold_1': {'start': 117768, 'end': 220406},
    'fold_2': {'start': 220407, 'end': 323045},
    'fold_3': {'start': 323046, 'end': 425684},
    'fold_4': {'start': 425685, 'end': 528323},
    'fold_5': {'start': 528324, 'end': 630964}
}
DATABASE_PATH = os.environ.get("DATABASE_PATH", '../database')
COCO_IMAGES_DIR_PATH = f'{DATABASE_PATH}/coco_images'