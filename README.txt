python preprocessing.py -imgs_dir /home/par/thesis/data/cc359/source/images -masks_dir /home/par/thesis/data/cc359/source/masks -patch_size_row 128 -patch_size_co 128


 python train.py -batch_size ?? -num_workers ??

  ./scripts/split-source-target.sh -r /home/par/thesis/data/cc359 -i /home/par/thesis/data/cc359/orig-sample -m /home/par/thesis/data/cc359/silver-standard-sample

   ./scripts/split-source-target.sh -r /home/parisaat/scratch/data/cc359 -i /home/parisaat/scratch/data/cc359/original -m /home/parisaat/scratch/data/cc359/silver_standard

   ENV=PRODUCTION python data/preprocessing.py -imgs_dir /home/parisaat/scratch/data/cc359/source/images -masks_dir /home/parisaat/scratch/data/cc359/source/masks


l226 /Users/par/PycharmProjects/Brain-MR-Segmentation-Playground/venv/lib/python3.8/site-packages/medicaltorch/datasets.py": input_data_shape[2] -> input_data_shape[self.slice_axis]
"/Users/par/PycharmProjects/Brain-MR-Segmentation-Playground/venv/lib/python3.8/site-packages/medicaltorch/datasets.py", line 13:
from torch._six import string_classes
int_classes = int


l226 /Users/par/PycharmProjects/Brain-MR-Segmentation-Playground/venv/lib/python3.8/site-packages/medicaltorch/datasets.py": input_data_shape[2] -> input_data_shape[self.slice_axis]
"/Users/par/PycharmProjects/Brain-MR-Segmentation-Playground/venv/lib/python3.8/site-packages/medicaltorch/datasets.py", line 13:
from torch._six import string_classes
int_classes = int