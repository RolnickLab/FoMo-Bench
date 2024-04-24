#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

full_path="$root_folder_path/SSL4EOS1S2"

mkdir $full_path

s2_url="https://dataserv.ub.tum.de/s/m1660427.001/download?path=%2Fssl4eo-s12&files=s2_l2a.tar.gz"

s1_url="https://dataserv.ub.tum.de/s/m1660427.001/download?path=%2Fssl4eo-s12&files=s1.tar.gz"

mkdir $full_path/"S2"
mkdir $full_path/"S1"

echo "Downloading Sentinel-1 from $s1_url"

wget --no-check-certificate -O "$full_path/s1.tar.gz" $s1_url

if [ $? -eq 0 ]; then
echo "Sentinel-1 downloading finished!"
else
echo "Downloading failed."
fi

echo "Extracting Sentinel-1"

tar -xvf "$full_path/s1.tar.gz" --directory "$full_path/S1";


if [ $? -eq 0 ]; then
echo "Sentinel-1 extraction finished!"
else
echo "Extraction failed."
fi



echo "Downloading Sentinel-2 from $s2_url"

wget --no-check-certificate -O "$full_path/s2.tar.gz" $s2_url

if [ $? -eq 0 ]; then
echo "Sentinel-2 downloading finished!"
else
echo "Downloading failed."
fi


echo "Extracting Sentinel-2"

tar -xvf "$full_path/s2.tar.gz" --directory "$full_path/S2";


if [ $? -eq 0 ]; then
echo "Sentinel-1 extraction finished!"
else
echo "Extraction failed."
fi
