#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

dataset_folder="FLAIR_2"

train_aerial_url=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2/flair_aerial_train.zip
train_sentinel_url=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2/flair_sen_train.zip
train_labels_url=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2/flair_labels_train.zip

test_aerial_url=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2/flair_2_aerial_test.zip
test_sentinel_url=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2/flair_2_sen_test.zip

aerial_meta_url=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2/flair_2_aerial_metadata.zip
aerial_sentinel_mapping=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2/flair_2_centroids_sp_to_patch.zip

urls=($train_aerial_url $train_sentinel_url $train_labels_url $test_aerial_url $test_sentinel_url $aerial_meta_url $aerial_sentinel_mapping)

full_path="$root_folder_path/$dataset_folder"

mkdir $full_path
for url in "${urls[@]}"
do
clear_name=$(basename "$url")
wget -O "$full_path/$clear_name" $url
if [ $? -eq 0 ]; then
echo "$clear_name Download complete. Unzipping!"
unzip "$full_path/$clear_name" -d "$full_path"
echo "$clear_name extracted. Removing zip file"
rm $full_path/$clear_name
else
echo "$clear_name Download failed."
fi
done
