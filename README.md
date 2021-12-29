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
```sh
./scripts/split-six-domains-ss.sh -r /path/to/data/cc359 -i /path/to/data/cc359/original -m /path/to/data/cc359/staple
./scripts/split-six-domains-wgc.sh -r /path/to/data/cc359 -i /path/to/data/cc359/orig -m /path/to/data/cc359/wgc
```