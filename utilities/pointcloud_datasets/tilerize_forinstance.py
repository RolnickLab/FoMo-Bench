import pandas as pd
from pathlib import Path
from utilities.tilerizer import Tilerizer


def main():
    # Please modify the folder_path to the NeonTree official dataset
    folder_path = "path/to/dataset"
    forinstance_path = Path(folder_path)
    paths = pd.read_csv(forinstance_path / "data_split_metadata.csv")["path"]
    paths = [forinstance_path / path for path in paths]
    for i, pc_path in enumerate(paths):
        print("Starting sub point cloud {}:".format(pc_path))
        tilerizer = Tilerizer(pc_path, task="segmentation", modality="point_cloud")
        samples = tilerizer.create_subpointcloud(nb_max_points=100000)


if __name__ == "__main__":
    main()
