python data/preprocessing.py -imgs_dir /home/par/thesis/data/cc359/source/images -masks_dir /home/par/thesis/data/cc359/source/masks -patch_size_row 128 -patch_size_co 128


 python train.py -batch_size ?? -num_workers ??

  ./scripts/split-source-target.sh -r /home/par/thesis/data/cc359 -i /home/par/thesis/data/cc359/orig -m /home/par/thesis/data/cc359/silver

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


python data/preprocessing.py -data_dir /home/parisaat/scratch/data/cc359/philips15

./scripts/split-six-domains.sh -r /home/parisaat/scratch/data/cc359 -i /home/parisaat/scratch/data/cc359/original -m /home/parisaat/scratch/data/cc359/staple

./scripts/split-source-wgc.sh -r /home/parisaat/scratch/data/cc359 -i /home/par/thesis/data/cc359/orig -m /home/parisaat/scratch/data/cc359/wgc

ENV=PRODUCTION python transfer_learnign.py -data_dir /home/parisaat/scratch/data/cc359/philips3/slices/train -model_name philips15_baseline -experiment_name tl_ph15_ph3

 ENV=PRODUCTION python mean_teacher.py experiments/train_ph15_adapt_ph3.json