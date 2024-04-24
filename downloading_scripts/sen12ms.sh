#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1
full_path="$root_folder_path/Sen12MS"
mkdir $full_path
download_link=https://dataserv.ub.tum.de/s/m1474000/download

wget --no-check-certificate -O "$full_path/sen12ms.zip" $download_link

if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

echo "Extracting data"

unzip "$full_path/sen12ms.zip" -d $full_path

if [ $? -eq 0 ]; then
echo "Removing zip files"
rm "$full_path/sen12ms.zip"
else
echo "Extraction failed."
fi

mv "$full_path/m1474000"/* "$full_path/"
rm -rf "$full_path/m1474000/"
mkdir "$full_path/ROIs"

for file in "$full_path"/*.tar.gz;
do
tar -xvf "$file" --directory "$full_path/ROIs";
rm "$file"
done

echo "Downloading train/test splits and labels"

wget -O "$full_path/test_list.txt" https://raw.githubusercontent.com/schmitt-muc/SEN12MS/master/splits/test_list.txt

if [ $? -eq 0 ]; then
echo ""
else
echo "Downloading failed."
fi

wget -O "$full_path/train_list.txt" https://raw.githubusercontent.com/schmitt-muc/SEN12MS/master/splits/train_list.txt

if [ $? -eq 0 ]; then
echo ""
else
echo "Downloading failed."
fi

wget -O "$full_path/IGBP_probability_labels.pkl" https://raw.githubusercontent.com/schmitt-muc/SEN12MS/master/labels/IGBP_probability_labels.pkl

if [ $? -eq 0 ]; then
echo "Downloading finished"
else
echo "Downloading failed."
fi
