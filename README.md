# Brain MRI Segmentation Playground


![](images/domain_adaptation_problem.png)

## Prerequisites 
- Linux
- Python 3.8
- PyTorch
- NVIDIA GPU

## Getting started
### Instalation
- Clone this repo
- Install the requirements

### Dataset
The CC-359 dataset is available at this [website](https://www.ccdataset.com/).

## Usage

### Preprocessing
If  you wish to use the same datasets as ours, you can use our preprocessing scripts. 

To split the dataset to six domains run the following scrips:

```sh
./scripts/split-six-domains-ss.sh -r /path/to/data/cc359 -i /path/to/data/cc359/original -m /path/to/data/cc359/staple
./scripts/split-six-domains-wgc.sh -r /path/to/data/cc359 -i /path/to/data/cc359/orig -m /path/to/data/cc359/wgc
```

Do the train and test split and preprocessing with following command for each data domain you use:

```sh
python data/preprocessing.py -data_dir /path/to/data/cc359/domain_name
```

### Training
To train a model from a method run the following command:
```sh
python mothods/method_name.py experiments/experimental_setup.json
```

### Testing

To test a model rub the following command:

```sh
python test.py -model_name model_name -experiment_name exp_name -data_dir /path/to/data/cc359/test
```