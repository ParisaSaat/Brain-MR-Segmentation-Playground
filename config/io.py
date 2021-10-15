import os

PROCESS_ENV = os.environ.get('ENV', 'TEST')


SOURCE_TRAIN_IMAGES_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/train/images',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/train/images',
}
SOURCE_TRAIN_IMAGES_PATH = SOURCE_TRAIN_IMAGES_PATH[PROCESS_ENV]

SOURCE_TRAIN_MASKS_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/train/masks',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/train/masks',
}
SOURCE_TRAIN_MASKS_PATH = SOURCE_TRAIN_MASKS_PATH[PROCESS_ENV]

SOURCE_VAL_IMAGES_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/val/images',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/val/images',
}
SOURCE_VAL_IMAGES_PATH = SOURCE_VAL_IMAGES_PATH[PROCESS_ENV]

SOURCE_VAL_MASKS_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/val/masks',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/val/masks',
}
SOURCE_VAL_MASKS_PATH = SOURCE_VAL_MASKS_PATH[PROCESS_ENV]

SOURCE_TEST_IMAGES_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/test/images',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/test/images',
}
SOURCE_TEST_IMAGES_PATH = SOURCE_TEST_IMAGES_PATH[PROCESS_ENV]

SOURCE_TEST_MASKS_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/test/masks',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/test/masks',
}
SOURCE_TEST_MASKS_PATH = SOURCE_TEST_MASKS_PATH[PROCESS_ENV]

SOURCE_SLICES_TRAIN_IMAGES_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/slices/train/images',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/slices/train/images',
}
SOURCE_SLICES_TRAIN_IMAGES_PATH = SOURCE_SLICES_TRAIN_IMAGES_PATH[PROCESS_ENV]

SOURCE_SLICES_TRAIN_MASKS_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/slices/train/masks',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/slices/train/masks',
}
SOURCE_SLICES_TRAIN_MASKS_PATH = SOURCE_SLICES_TRAIN_MASKS_PATH[PROCESS_ENV]

SOURCE_SLICES_VAL_IMAGES_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/slices/val/images',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/slices/val/images',
}
SOURCE_SLICES_VAL_IMAGES_PATH = SOURCE_SLICES_VAL_IMAGES_PATH[PROCESS_ENV]

SOURCE_SLICES_VAL_MASKS_PATH = {
    'TEST': '/home/par/thesis/data/cc359/source/slices/val/masks',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/source/slices/val/masks',
}
SOURCE_SLICES_VAL_MASKS_PATH = SOURCE_SLICES_VAL_MASKS_PATH[PROCESS_ENV]

NPY_ROOT = {
    'TEST': '/home/par/thesis/data/cc359/npy',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/npy',
}
NPY_ROOT = NPY_ROOT[PROCESS_ENV]

NORMALIZED_IMAGES_PATH = {
    'TEST': '/home/par/thesis/data/cc359/normalized',
    'PRODUCTION': '/home/parisaat/scratch/data/cc359/normalized',
}
NORMALIZED_IMAGES_PATH = NORMALIZED_IMAGES_PATH[PROCESS_ENV]

IMAGE_FILE_TEMPLATE = '{file_id}.nii.gz'
MASK_FILE_TEMPLATE = '{file_id}_ss.nii.gz'
NORMALIZED_IMAGES_TEMPLATE = '{file_id}.nii'
