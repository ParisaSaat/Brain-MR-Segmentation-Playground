#!/bin/bash

while getopts r:i:m: flag
do
    case "${flag}" in
        r) root_path=${OPTARG};;
        i) images_path=${OPTARG};;
        m) masks_path=${OPTARG};;
    esac
done


mkdir -p $root_path/philips15/images
mkdir -p $root_path/philips15/masks
mkdir -p $root_path/siemens15/images
mkdir -p $root_path/siemens15/masks
mkdir -p $root_path/ge15/images
mkdir -p $root_path/ge15/masks
mkdir -p $root_path/philips3/images
mkdir -p $root_path/philips3/masks
mkdir -p $root_path/siemens3/images
mkdir -p $root_path/siemens3/masks
mkdir -p $root_path/ge3/images
mkdir -p $root_path/ge3/masks



cp $images_path/*_philips_15_* $root_path/philips15/images
cp $masks_path/*_philips_15_* $root_path/philips15/masks
cp $images_path/*_siemens_15_* $root_path/siemens15/images
cp $masks_path/*_siemens_15_* $root_path/siemens15/masks
cp $images_path/*_ge_15_* $root_path/ge15/images
cp $masks_path/*_ge_15_* $root_path/ge15/masks
cp $images_path/*_philips_3_* $root_path/philips3/images
cp $masks_path/*_philips_3_* $root_path/philips3/masks
cp $images_path/*_siemens_3_* $root_path/siemens3/images
cp $masks_path/*_siemens_3_* $root_path/siemens3/masks
cp $images_path/*_ge_3_* $root_path/ge3/images
cp $masks_path/*_ge_3_* $root_path/ge3/masks
