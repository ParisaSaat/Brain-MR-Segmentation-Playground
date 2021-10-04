from os import path

SOURCE_TRAIN_IMAGES_PATH = '/home/parisaat/scratch/benchmark_data/cc359/source/train/images'
SOURCE_TRAIN_MASKS_PATH = '/home/parisaat/scratch/benchmark_data/cc359/source/train/masks'
SOURCE_TEST_IMAGES_PATH = '/home/parisaat/scratch/benchmark_data/cc359/source/test/images'
SOURCE_TEST_MASKS_PATH = '/home/parisaat/scratch/benchmark_data/cc359/source/test/masks'
IMAGE_FILE_TEMPLATE = '{file_id}.nii.gz'
MASK_FILE_TEMPLATE = '{file_id}_ss.nii.gz'
NPY_ROOT = '/home/parisaat/scratch/benchmark_data/cc359/npy'
TRAIN_IMAGES_PATCHES_PATH = path.join(NPY_ROOT, 'source_train_images_patches.npy')
TRAIN_MASKS_PATCHES_PATH = path.join(NPY_ROOT, 'source_train_masks_patches.npy')
TEST_IMAGES_PATCHES_PATH = path.join(NPY_ROOT, 'source_test_images_patches.npy')
TEST_MASKS_PATCHES_PATH = path.join(NPY_ROOT, 'source_test_masks_patches.npy')
NORMALIZED_IMAGES_PATH = '/home/parisaat/scratch/benchmark_data/cc359/normalized'
NORMALIZED_IMAGES_TEMPLATE = '{file_id}.nii'
