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

MODEL_PATH = {
    'TEST': '/home/par/PycharmProjects/Brain-MR-Segmentation-Playground/models/{model_name}.pth',
    'PRODUCTION': '/home/muhammadathar.ganaie/Brain-MR-Segmentation-Playground/trained_models/{model_name}.pth',
}
MODEL_PATH = MODEL_PATH[PROCESS_ENV]

IMAGE_FILE_TEMPLATE = '{file_id}.nii.gz'
MASK_FILE_TEMPLATE = '{file_id}_{mask_type}.nii.gz'
NORMALIZED_IMAGES_TEMPLATE = '{file_id}.nii'
LOSS_PATH = 'losses'
PRETRAIN_UNET = 'pretrain_unet'
PRETRAIN_SEGMENTER = 'pretrain_segmenter'
PRETRAIN_DOMAIN = 'pretrain_domain'
CHK_PATH_UNET = 'unet_pth_checkpoint'
CHK_PATH_SEGMENTER = 'segmenter_pth_checkpoint'
CHK_PATH_DOMAIN = 'domain_pth_checkpoint'
PATH_UNET = 'unet_pth'
PATH_SEGMENTER = 'segmenter_pth'
PATH_DOMAIN = 'domain_pth'
