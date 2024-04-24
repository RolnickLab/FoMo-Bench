import os
import pickle
import pprint
import random
import warnings

from pathlib import Path
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjson5 as json
import rasterio
import torch
import xmltodict
from tqdm import tqdm

from torch_geometric.data import Data
import torch_geometric.transforms as T

import utilities.augmentations

"""
Data loading for the FORinstance Dataset published in:
https://zenodo.org/record/8287792
"""


class FORinstanceDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        self.mode = mode
        self.configs = configs
        self.root_path = Path(configs["root_path"])
        self.seg_task = self.configs["segmentation_task"]
        self.samples = self._load_dataset()

        if "nb_points" in self.configs.keys():
            self.nb_points = self.configs["nb_points"]
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None
        self.normalization = T.NormalizeScale()

        # debug
        # self.samples = self.samples[:100]

        if self.mode == "train":
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[: int(0.9 * len(self.samples))]

        elif self.mode == "val":
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[int(0.9 * len(self.samples)) :]

        self.num_examples = len(self.samples)
        print("Number of samples in split {}  = {}".format(self.mode, self.num_examples))

    def _load_dataset(self):
        dataset = pd.read_csv(self.root_path / "data_split_metadata.csv")
        if self.mode in ("train", "val"):
            dataset = dataset[dataset["split"] == "dev"]
        else:
            dataset = dataset[dataset["split"] == "test"]
        if self.seg_task == "semantic_segmentation":
            dataset = dataset[dataset["folder"] != "RMIT"]
            dataset = dataset[dataset["folder"] != "TUWIEN"]
        path_files = [
            self.root_path / sample_path for sample_path in dataset["path"] if (self.root_path / sample_path).exists()
        ]
        path_files = [list(path_file.parent.glob(path_file.stem + "*.pkl")) for path_file in path_files]
        path_files = [item for sublist in path_files for item in sublist]
        samples = []

        for path_file in path_files:
            with open(path_file, "rb") as f:
                samples.append(pickle.load(f))
        samples = [item["sub_pc"] for sublist in samples for item in sublist]
        return samples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        sample = self.samples[index]
        with laspy.open(sample) as lidar_file:
            pc = lidar_file.read()
        if self.seg_task == "semantic_segmentation":
            # labels = np.array(pc.treeSP, dtype=np.int64)
            labels = np.array(pc.classification, dtype=np.int64)
        elif self.seg_task == "instance_segmentation":
            labels = np.array(pc.treeID, dtype=np.int64)
        else:
            raise Exception("Task {} is not supported yet.".format(self.seg_task))
        point_cloud = np.stack([np.array(pc.x), np.array(pc.y), np.array(pc.z)]).T
        # get random sub sample
        if self.mode == "train":
            if self.nb_points > point_cloud.shape[0]:
                replace = True
            else:
                replace = False
            rand_idx = np.random.choice(list(range(point_cloud.shape[0])), size=self.nb_points, replace=replace)
            point_cloud = point_cloud[rand_idx, :]
            labels = labels[rand_idx]
        point_cloud = torch.tensor(point_cloud)
        labels = torch.tensor(labels)
        # No feature is considered for baselines
        x = torch.ones((point_cloud.shape[0], 3), dtype=torch.float)
        data = Data(pos=point_cloud, x=x, y=labels)
        data = self.normalization(data)
        if self.configs["augment"] and self.mode == "train":
            data = self.augmentations(data)
        return data, None

    def collate_fn(self, batch):
        return tuple(zip(*batch))


if __name__ == "__main__":
    dataset = FORinstanceDataset(path_augment_dict=True, mode="train")
    for i, data in enumerate(dataset):
        import ipdb

        ipdb.set_trace()
    # dataset.plot(0)
