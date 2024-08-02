import os
import sys
from pathlib import Path

# Allow loading tilerizer from the above hierarchy
higher_level_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(higher_level_folder)

from tilerizer import Tilerizer


def tile_dataset(tile_paths, annot_path):
    for rgb_file_path in tile_paths:
        file_name = rgb_file_path.stem
        print("Starting tiling RGB file: {}".format(file_name))
        tilerizer = Tilerizer(rgb_file_path, annot_path)
        tilerizer.create_tiles(tile_size=1000)


def main():
    # Please modify the folder_path to the NeonTree official dataset
    folder_path = "path/to/dataset"
    reforestree_path = Path(folder_path)
    annot_path = reforestree_path / "mapping" / "final_dataset.csv"
    rgb_data_paths = list((reforestree_path / "tiles").glob("*/*.png"))
    tile_dataset(rgb_data_paths, annot_path)


if __name__ == "__main__":
    main()
