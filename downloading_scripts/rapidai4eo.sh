#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

full_path="$root_folder_path"

mkdir $full_path

azcopy copy --recursive https://radiantearth.blob.core.windows.net/mlhub/rapidai4eo/ "$full_path"