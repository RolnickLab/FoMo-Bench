#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

#Urls to download
pinus_url="https://zenodo.org/record/7565500/files/data_pinus_radiata.zip?download=1"
europaeus_url="https://zenodo.org/record/7565490/files/data_ulex_europaeus.zip?download=1"
acacia_url="https://zenodo.org/record/7565546/files/data_acacia_dealbata.zip?download=1"

#Filenames for each url
filenames=("data_pinus_radiata.zip" "data_ulex_europaeus.zip" "data_acacia_dealbata.zip")
urls=($pinus_url $europaeus_url $acacia_url)

#Create folder if it doesn't exist
dataset_folder="Woody"
mkdir $root_folder_path/$dataset_folder

#Download each url
full_path="$root_folder_path/$dataset_folder"
for ((index=0;index<${#filenames[@]}; index++))
do
    echo "Saving to $full_path/"${filenames[index]}""
    clean_name=${filenames[index]%.zip}
    # Use the curl command to download the file
    wget -O "$full_path"/"${filenames[index]}"  "${urls[index]}" 
    unzip "$full_path"/"${filenames[index]}" -d "$full_path"
    rm "$full_path"/"${filenames[index]}"
    echo "File "${filenames[index]}" downloaded successfully!"
done
# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Download complete."
else
echo "Download failed."
fi
