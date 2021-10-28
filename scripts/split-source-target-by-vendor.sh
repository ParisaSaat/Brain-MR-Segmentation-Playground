#!/bin/bash

while getopts r:i:m: flag
do
    case "${flag}" in
        r) root_path=${OPTARG};;
        i) images_path=${OPTARG};;
        m) masks_path=${OPTARG};;
    esac
done


mkdir -p $root_path/philips/images
mkdir -p $root_path/philips/masks
mkdir -p $root_path/siemens/images
mkdir -p $root_path/siemens/masks
mkdir -p $root_path/ge/images
mkdir -p $root_path/ge/masks


cp $images_path/*_philips_* $root_path/philips/images
cp $masks_path/*_philips_* $root_path/philips/masks
cp $images_path/*_siemens_* $root_path/siemens/images
cp $masks_path/*_siemens_* $root_path/siemens/masks
cp $images_path/*_ge_* $root_path/ge/images
cp $masks_path/*_ge_* $root_path/ge/masks
