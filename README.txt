python preprocessing.py -imgs_dir /Users/par/Desktop/uni/thesis/data/cc359/source/images -masks_dir
/Users/par/Desktop/uni/thesis/data/cc359/source/masks -patch_size_row 128 -patch_size_co 128


 python train.py -batch_size ?? -num_workers ??

  ./scripts/split-source-target.sh -r /Users/par/Desktop/uni/thesis/data/cc359 -i
  /Users/par/Desktop/uni/thesis/data/cc359/orig-sample -m
  /Users/par/Desktop/uni/thesis/data/cc359/Silver-standard-sample

   ./scripts/split-source-target.sh -r /home/parisaat/scratch/benchmark_data/cc359 -i /home/parisaat/scratch/benchmark_data/cc359/original -m /home/parisaat/scratch/benchmark_data/cc359/silver_standard

   python preprocessing.py -imgs_dir /home/parisaat/scratch/benchmark_data/cc359/source/images -masks_dir /home/parisaat/scratch/benchmark_data/cc359/source/masks -patch_size_row 128 -patch_size_co 128
