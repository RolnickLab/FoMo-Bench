#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi

root_folder_path=$1

full_path="$root_folder_path/BigEarthNet"

mkdir $full_path

sentinel2_url=https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz
sentinel1_url=https://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz
label_url=https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/raw/master/label_indices.json?inline=false

train_url=https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/raw/master/splits/train.csv?inline=false
val_url=https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/raw/master/splits/val.csv?inline=false
test_url=https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/raw/master/splits/test.csv?inline=false

echo "Downloading Sentinel-2 data"

wget -O "$full_path/BigEarthNet-S2-v1.0.tar.gz" $sentinel2_url


# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

echo "Downloading Sentinel-1 data"

wget -O "$full_path/BigEarthNet-S1-v1.0.tar.gz" $sentinel1_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

echo "Downloading metadata"

wget -O "$full_path/label_indices.json" $label_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

wget -O "$full_path/train.csv" $train_url
# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

wget -O "$full_path/val.csv" $val_url
# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

wget -O "$full_path/test.csv" $test_url
# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

wget -O "$full_path/s1s2_mapping.csv" https://git.tu-berlin.de/rsim/BigEarthNet-MM_tools/-/raw/master/files/s1s2_mapping.csv?inline=false
# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

echo "Untaring data"

tar -xvf "$full_path/BigEarthNet-S2-v1.0.tar.gz" --directory $full_path

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Extracted Sentinel-2 data succesfully."
else
echo "Sentinel-2 extraction failed."
fi


tar -xvf "$full_path/BigEarthNet-S1-v1.0.tar.gz" --directory $full_path

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Extracted Sentinel-1 data succesfully."
else
echo "Sentinel-1 extraction failed."
fi


echo "Removing tar files"

rm "$full_path/BigEarthNet-S2-v1.0.tar.gz"
rm "$full_path/BigEarthNet-S1-v1.0.tar.gz"

echo "Download finished."