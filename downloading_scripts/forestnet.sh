#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

download_link=http://download.cs.stanford.edu/deep/ForestNetDataset.zip

dataset_folder="ForestNet"
filename="ForestNet.zip"
full_path="$root_folder_path/$dataset_folder"

mkdir $full_path

wget -O "$full_path/$filename" $download_link

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "ForestNet download finished normally."
else
echo "Downloading failed."
fi

echo "Extracting dataset"

unzip "$full_path/$filename" -d "$full_path"

rm "$full_path/$filename"

sub_dir="/deep/downloads/ForestNetDataset"
mv "$full_path/$sub_dir"/* $full_path
rm -rf "$full_path/deep"
echo "ForestNet has been extracted."