import yaml
from pathlib import Path
from utilities.tilerizer import Tilerizer


def main():
    # Please modify the folder_path to the NeonTree official dataset
    folder_path = "path/to/dataset"
    neontree_path = Path(folder_path)
    with open(neontree_path / "lidar_annots_paths.yml", "r") as fp:
        paths = yaml.safe_load(fp)
    for split in ("training", "evaluation"):
        for i, pc_path in enumerate(paths[split]["annot_laz"]):
            pc_path = Path(pc_path)
            print("Starting sub point cloud {}:".format(pc_path))
            tilerizer = Tilerizer(pc_path, task="segmentation", modality="point_cloud")
            samples = tilerizer.create_subpointcloud(nb_max_points=100000)


if __name__ == "__main__":
    main()
