from os import path

SOURCE_TRAIN_IMAGES_PATH = {
    'test': '/home/par/thesis/data/cc359/source/train/images',
    'production': '/home/parisaat/scratch/data/cc359/source/train/images',
}

SOURCE_TRAIN_MASKS_PATH = {
    'test': '/home/par/thesis/data/cc359/source/train/masks',
    'production': '/home/parisaat/scratch/data/cc359/source/train/masks',
}

SOURCE_VAL_IMAGES_PATH = {
    'test': '/home/par/thesis/data/cc359/source/val/images',
    'production': '/home/parisaat/scratch/data/cc359/source/val/images',
}
SOURCE_VAL_MASKS_PATH = {
    'test': '/home/par/thesis/data/cc359/source/val/masks',
    'production': '/home/parisaat/scratch/data/cc359/source/val/masks',
}
SOURCE_TEST_IMAGES_PATH = {
    'test': '/home/par/thesis/data/cc359/source/test/images',
    'production': '/home/parisaat/scratch/data/cc359/source/test/images',
}
SOURCE_TEST_MASKS_PATH = {
    'test': '/home/par/thesis/data/cc359/source/test/masks',
    'production': '/home/parisaat/scratch/data/cc359/source/test/masks',
}
SOURCE_SLICES_TRAIN_IMAGES_PATH = {
    'test': '/home/par/thesis/data/cc359/source/slices/train/images',
    'production': '/home/parisaat/scratch/data/cc359/source/slices/train/images',
}
SOURCE_SLICES_TRAIN_MASKS_PATH = {
    'test': '/home/par/thesis/data/cc359/source/slices/train/masks',
    'production': '/home/parisaat/scratch/data/cc359/source/slices/train/masks',
}
SOURCE_SLICES_VAL_IMAGES_PATH = {
    'test': '/home/par/thesis/data/cc359/source/slices/val/images',
    'production': '/home/parisaat/scratch/data/cc359/source/slices/val/images',
}
SOURCE_SLICES_VAL_MASKS_PATH = {
    'test': '/home/par/thesis/data/cc359/source/slices/val/masks',
    'production': '/home/parisaat/scratch/data/cc359/source/slices/val/masks',
}

NPY_ROOT = {
    'test': '/home/par/thesis/data/cc359/npy',
    'production': '/home/parisaat/scratch/data/cc359/npy',
}

TRAIN_IMAGES_PATCHES_PATH = path.join(NPY_ROOT, 'source_train_images_patches.npy')
TRAIN_MASKS_PATCHES_PATH = path.join(NPY_ROOT, 'source_train_masks_patches.npy')

VAL_IMAGES_PATCHES_PATH = path.join(NPY_ROOT, 'source_val_images_patches.npy')
VAL_MASKS_PATCHES_PATH = path.join(NPY_ROOT, 'source_val_masks_patches.npy')

TEST_IMAGES_PATCHES_PATH = path.join(NPY_ROOT, 'source_test_images_patches.npy')
TEST_MASKS_PATCHES_PATH = path.join(NPY_ROOT, 'source_test_masks_patches.npy')
NORMALIZED_IMAGES_PATH = {
    'test': '/home/par/thesis/data/cc359/normalized',
    'production': '/home/parisaat/scratch/data/cc359/normalized',
}

IMAGE_FILE_TEMPLATE = '{file_id}.nii.gz'
MASK_FILE_TEMPLATE = '{file_id}_ss.nii.gz'
NORMALIZED_IMAGES_TEMPLATE = '{file_id}.nii'
