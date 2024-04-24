#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

aerial_albies_alba=https://zenodo.org/record/6778154/files/aerial_60m_abies_alba.zip?download=1
aerial_acer_pseudoplatanus=https://zenodo.org/record/6778154/files/aerial_60m_acer_pseudoplatanus.zip?download=1
aerial_alnus_spec=https://zenodo.org/record/6778154/files/aerial_60m_alnus_spec.zip?download=1
aerial_betula_spec=https://zenodo.org/record/6778154/files/aerial_60m_betula_spec.zip?download=1
aerial_cleared=https://zenodo.org/record/6778154/files/aerial_60m_cleared.zip?download=1
aerial_fagus_sylvatica=https://zenodo.org/record/6778154/files/aerial_60m_fagus_sylvatica.zip?download=1
aerial_fraxinus_excelsior=https://zenodo.org/record/6778154/files/aerial_60m_fraxinus_excelsior.zip?download=1
aerial_larix_decidua=https://zenodo.org/record/6778154/files/aerial_60m_larix_decidua.zip?download=1
aerial_larix_kaempferi=https://zenodo.org/record/6778154/files/aerial_60m_larix_kaempferi.zip?download=1
aerial_picea_albies=https://zenodo.org/record/6778154/files/aerial_60m_picea_abies.zip?download=1
aerial_pinus_nigra=https://zenodo.org/record/6778154/files/aerial_60m_pinus_nigra.zip?download=1
aerial_pinus_strobus=https://zenodo.org/record/6778154/files/aerial_60m_pinus_strobus.zip?download=1
aerial_pinus_sylvestris=https://zenodo.org/record/6778154/files/aerial_60m_pinus_sylvestris.zip?download=1
aerial_populus_spec=https://zenodo.org/record/6778154/files/aerial_60m_populus_spec.zip?download=1
aerial_prunus_spec=https://zenodo.org/record/6778154/files/aerial_60m_prunus_spec.zip?download=1 
aerial_pseudotsuga=https://zenodo.org/record/6778154/files/aerial_60m_pseudotsuga_menziesii.zip?download=1
aerial_quercus_petraea=https://zenodo.org/record/6778154/files/aerial_60m_quercus_petraea.zip?download=1
aerial_quercus_robur=https://zenodo.org/record/6778154/files/aerial_60m_quercus_robur.zip?download=1
aerial_quercus_rubra=https://zenodo.org/record/6778154/files/aerial_60m_quercus_rubra.zip?download=1
aerial_tilia_spec=https://zenodo.org/record/6778154/files/aerial_60m_tilia_spec.zip?download=1

geo_json_url=https://zenodo.org/record/6778154/files/geojson.zip?download=1
labels_url=https://zenodo.org/record/6778154/files/labels.zip?download=1

s1_url=https://zenodo.org/record/6778154/files/s1.zip?download=1
s2_url=https://zenodo.org/record/6778154/files/s2.zip?download=1

test_filenames_url=https://zenodo.org/record/6778154/files/test_filenames.lst?download=1
train_filenames_url=https://zenodo.org/record/6778154/files/train_filenames.lst?download=1




aerial_urls=($aerial_albies_alba $aerial_acer_pseudoplatanus $aerial_alnus_spec $aerial_betula_spec $aerial_cleared $aerial_fagus_sylvatica $aerial_fraxinus_excelsior $aerial_larix_decidua $aerial_larix_kaempferi $aerial_picea_albies $aerial_pinus_nigra $aerial_pinus_strobus $aerial_pinus_sylvestris $aerial_populus_spec $aerial_prunus_spec $aerial_pseudotsuga $aerial_quercus_petraea $aerial_quercus_robur $aerial_quercus_rubra $aerial_tilia_spec)
#Filenames for each url
aerial_filenames=("albies_alba.zip" "acer_pseudoplatanus.zip" "alnus_spec.zip" "betula_spec.zip" "cleared.zip" "fagus_sylvatica.zip" "fraxinus_excelsior.zip" "larix_decidua.zip" "larix_kaempferi.zip" "picea_albies.zip" "pinus_nigra.zip" "pinus_strobus.zip" "pinus_sylvestris.zip" "populus_spec.zip" "prunus_spec.zip" "pseudotsuga_menziesii.zip" "quercus_petraea.zip" "quercus_robur" "quercus_rubra.zip" "tilia_spec.zip")

#Create folder if it doesn't exist
dataset_folder="TreeSat"
mkdir $root_folder_path/$dataset_folder

mkdir  $root_folder_path/$dataset_folder/"aerial_60m" $root_folder_path/$dataset_folder/"geojson" $root_folder_path/$dataset_folder/"labels"
echo "Downloading aerial data."
#Download each url
full_path="$root_folder_path/$dataset_folder"

for ((index=0;index<${#aerial_filenames[@]}; index++))
do
    file="aerial_60m/${aerial_filenames[index]}"
    echo "Saving to $full_path/$file"
    # Use the curl command to download the file
    wget -O "$full_path"/"$file"  "${aerial_urls[index]}" 
    clear_name=${file%.zip}
    mkdir "$full_path"/"$clear_name"
    unzip "$full_path"/"$file" -d "$full_path"/"$clear_name"
    rm "$full_path"/"$file"
    echo "File "${aerial_filenames[index]}" downloaded and extracted successfully!"
done

echo "Downloading Sentinel data"

sentinel_urls=($s1_url $s2_url)
sentinel_filenames=("s1.zip" "s2.zip")

for ((index=0;index<${#sentinel_urls[@]}; index++))
do
    current_name=${sentinel_filenames[index]}
    clear_name=${current_name%.zip}
    file="${sentinel_filenames[index]}"
    echo "Saving to $full_path/$file"
    # Use the wget command to download the file
    wget -O "$full_path/$file"  "${sentinel_urls[index]}" 
    unzip "$full_path/$file" -d "$full_path"
    rm "$full_path/$file"
    echo "File "${sentinel_filenames[index]}" downloaded and extracted successfully!"
done



echo "Downloading labels"

wget -O "$full_path/labels.zip" $labels_url
unzip "$full_path/labels.zip" -d "$full_path"
rm "$full_path/labels.zip"

echo "Downloading geojson metadata"
wget -O "$full_path/geojson.zip" $geo_json_url
unzip "$full_path/geojson.zip" -d "$full_path"
rm "$full_path/geojson.zip"


echo "Downloading train/test split filenames"

wget -O "$full_path/train_filenames.lst" $train_filenames_url
wget -O "$full_path/test_filenames.lst" $test_filenames_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

echo "Process finished. Exiting!"