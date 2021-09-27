#!/bin/bash

while getopts r:i:m: flag
do
    case "${flag}" in
        r) root_path=${OPTARG};;
        i) images_path=${OPTARG};;
        m) masks_path=${OPTARG};;
    esac
done


mkdir -p $root_path/source/images
mkdir -p $root_path/source/masks
mkdir -p $root_path/target/images
mkdir -p $root_path/target/masks


cp $images_path/*_15_* $root_path/source/images
cp $masks_path/*_15_* $root_path/source/masks
cp $images_path/*_3_* $root_path/target/images
cp $masks_path/*_3_* $root_path/target/masks

