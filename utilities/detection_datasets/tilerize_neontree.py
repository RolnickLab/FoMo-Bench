from pathlib import Path
from utilities.tilerizer import Tilerizer


def tile_training(tile_path, annot_path):
    tif_file_paths = tile_path.glob("*.tif")
    for tif_file_path in tif_file_paths:
        file_name = tif_file_path.stem
        annot_file_name = file_name + ".xml"
        annot_file_path = annot_path / annot_file_name
        print("Starting tiling TIF: {}".format(file_name))
        tilerizer = Tilerizer(tif_file_path, annot_file_path)
        tilerizer.create_tiles()


def tile_evaluation(tile_path, annot_path):
    tif_file_paths = tile_path.glob("*.tif")
    for tif_file_path in tif_file_paths:
        file_name = tif_file_path.stem
        annot_file_name = file_name + ".xml"
        annot_file_path = annot_path / annot_file_name
        if annot_file_path.exists():
            print("Starting tiling TIF: {}".format(file_name))
            tilerizer = Tilerizer(tif_file_path, annot_file_path)
            tilerizer.create_tiles()
        else:
            print("Following TIF file has no annotation: {}".format(file_name))


def main():
    # Please modify the folder_path to the NeonTree official dataset
    folder_path = "path/to/dataset"
    neontree_path = Path(folder_path)
    annot_path = neontree_path / "annotations"
    evaluation_rgb_path = neontree_path / "evaluation" / "RGB"
    training_rgb_path = neontree_path / "training" / "RGB"
    tile_training(training_rgb_path, annot_path)
    tile_evaluation(evaluation_rgb_path, annot_path)


if __name__ == "__main__":
    main()
