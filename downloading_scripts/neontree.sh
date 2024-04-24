#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

dataset_folder="NeonTree"
full_path="$root_folder_path/$dataset_folder"

mkdir $full_path
annotations_url=https://zenodo.org/record/5914554/files/annotations.zip?download=1
training_url=https://zenodo.org/record/5914554/files/training.zip?download=1
evaluation_url=https://zenodo.org/record/5914554/files/evaluation.zip?download=1

echo "Downloading files"

wget -O "$full_path/annotations.zip" $annotations_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "NeonTree download finished normally."
else
echo "Downloading failed."
fi

wget -O "$full_path/training.zip" $training_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "NeonTree download finished normally."
else
echo "Downloading failed."
fi

wget -O "$full_path/evaluation.zip" $evaluation_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "NeonTree download finished normally."
else
echo "Downloading failed."
fi
echo "Extracting files"

unzip "$full_path/annotations.zip" -d "$full_path"
mkdir "$full_path/training"
unzip "$full_path/training.zip" -d "$full_path/training/"
unzip "$full_path/evaluation.zip" -d "$full_path"

echo "Removing zip files"

rm "$full_path/annotations.zip" 
rm "$full_path/training.zip"
rm "$full_path/evaluation.zip"

rm -rf "$full_path/__MACOSX"