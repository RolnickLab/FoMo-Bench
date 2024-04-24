#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

train_url=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_1/flair_aerial_train.zip
test_url=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_1/flair_1_aerial_test.zip
labels_train=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_1/flair_labels_train.zip
labels_test=https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_1/flair_1_labels_test.zip

dataset_folder="FLAIR"
train_filename="train.zip"
test_filename="test.zip"
full_path="$root_folder_path/$dataset_folder"

mkdir $full_path

echo "Downloading training set."
wget -O "$full_path/$train_filename" $train_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Flair training set download finished normally."
else
echo "Downloading failed."
fi


echo "Downloading labels for the training set."
wget -O "$full_path/train_labels.zip" $labels_train

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Flair training labels download finished normally."
else
echo "Downloading failed."
fi


wget -O "$full_path/$test_filename" $test_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Flair training set download finished normally."
else
echo "Downloading failed."
fi

echo "Downloading labels for the test set."
wget -O "$full_path/test_labels.zip" $labels_test

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Flair test labels download finished normally."
else
echo "Downloading failed."
fi


echo "Extracting dataset"

unzip "$full_path/$train_filename" -d "$full_path"
unzip "$full_path/$test_filename" -d "$full_path"

unzip "$full_path/train_labels.zip" -d "$full_path"
unzip "$full_path/test_labels.zip" -d "$full_path"

rm "$full_path/$train_filename"
rm "$full_path/$test_filename"
rm "$full_path/train_labels.zip"
rm "$full_path/test_labels.zip"

echo "FLAIR extracting finished."