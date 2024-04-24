#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

wildforest_url=https://raw.githubusercontent.com/ekalinicheva/multi_layer_vegetation/main/DATASET/WildForest3D.zip
abbreviations_url=https://raw.githubusercontent.com/ekalinicheva/multi_layer_vegetation/main/DATASET/Abbreviations_species.xlsx

#Filenames for each url
filenames=("wildforest.zip" "abbreviations.zip")
urls=($wildforest_url $abbreviations_url)

#Create folder if it doesn't exist

#Download each url
full_path=$root_folder_path/"WildForest"
mkdir $full_path

echo "Downloading WildForest3D!"
wget -O "$full_path/wildforest.zip" $wildforest_url
unzip "$full_path/wildforest.zip" -d $full_path
rm "$full_path/wildforest.zip"

echo "WildForest3D downloaded and extracted successfully!"

echo "Downloading abbreviations!"
wget -O "$full_path/Abbreviations_species.xlsx" $abbreviations_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

echo "Process finished. Exiting!"