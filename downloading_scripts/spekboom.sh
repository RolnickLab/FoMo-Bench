if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

dataset_folder="Spekboom"
full_path="$root_folder_path/$dataset_folder"

mkdir $full_path

download_url=https://zenodo.org/record/7564954/files/data_spekboom.zip?download=1#!/bin/bash


echo "Downloading files"

wget -O "$full_path/spekboom.zip" $download_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Spekboom download finished normally."
else
echo "Downloading failed."
fi

echo "Extracting data"

unzip "$full_path/spekboom.zip" -d "$full_path"

echo "Removing zip files"

rm "$full_path/spekboom.zip"

