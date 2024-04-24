import pickle
import random

from pathlib import Path
import einops
import laspy
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torchvision import transforms

import albumentations as A
from albumentations.augmentations import Resize
from torch_geometric.data import Data
import torch_geometric.transforms as T

import utilities.augmentations
from utilities.utils import format_bboxes_voc_to_yolo
from datasets.NeonTreeDataset import NeonTreeDataset
from datasets.FORinstanceDataset import FORinstanceDataset


class MixedPCSegDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train", test_on="forinstance"):
        self.configs = configs
        self.mode = mode
        self.test_on = test_on
        self.dataset_names = self.configs["dataset_names"].split(",")
        self.nb_datasets = len(self.dataset_names)
        self.seg_task = self.configs["segmentation_task"]
        self.datasets = self._load_datasets()
        self.samples = self._group_samples()
        # This class support only pc data for the moment
        self.modality = "lidar"

        if "nb_points" in self.configs.keys():
            self.nb_points = self.configs["nb_points"]
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None
        self.normalization = T.NormalizeScale()

        # Full debugging
        # if self.mode in ('train', 'val'):
        # Debugging train/val
        # self.samples = self.samples[:100]
        # else:
        # Debugging test
        # self.samples['neontree_point_cloud'] = self.samples['neontree_point_cloud'][:100]
        # self.samples['forinstance'] = self.samples['forinstance'][:100]

        if self.mode == "test":
            self.num_examples = len(self.samples[self.test_on])
            # sum([len(self.samples[dataset]) for dataset in self.samples.keys()])
        else:
            self.num_examples = len(self.samples)
        print(
            "Total number of mixed samples in split "
            + "{} with modality: {}  = {}".format(self.mode, self.modality, self.num_examples)
        )

    def __len__(self):
        return self.num_examples

    def _load_datasets(self):
        datasets = dict()
        for dataset_name in self.dataset_names:
            if dataset_name.lower() == "neontree_point_cloud":
                self.configs["dataset_name"] = dataset_name.lower()
                self.configs["root_path"] = self.configs["root_path_neontree"]
                datasets[dataset_name.lower()] = NeonTreeDataset(self.configs, self.mode)
            if dataset_name.lower() == "forinstance":
                self.configs["dataset_name"] = dataset_name.lower()
                self.configs["root_path"] = self.configs["root_path_forinstance"]
                datasets[dataset_name.lower()] = FORinstanceDataset(self.configs, self.mode)
            if len(datasets) == 0:
                raise Exception("Dataset {} not supported".format(dataset_name))
        return datasets

    def _group_samples(self):
        if self.mode in ("train", "val"):
            samples = []
            for dataset_name in self.datasets.keys():
                sub_samples = self.datasets[dataset_name].samples
                # Need to add dataset name to each sample
                if dataset_name == "neontree_point_cloud":
                    sub_samples = [{"sub_pc": sample["sub_pc"], "dataset": dataset_name} for sample in sub_samples]
                else:
                    sub_samples = [{"sub_pc": sample, "dataset": dataset_name} for sample in sub_samples]
                samples += sub_samples
            random.Random(999).shuffle(samples)
        else:  # Supposed to be test
            samples = dict()
            for dataset_name in self.datasets.keys():
                sub_samples = self.datasets[dataset_name].samples
                # Need to add dataset name to each sample
                if dataset_name == "neontree_point_cloud":
                    sub_samples = [sample.update({"dataset": dataset_name}) for sample in sub_samples]
                else:
                    sub_samples = [{"sub_pc": sample, "dataset": dataset_name} for sample in sub_samples]
                samples[dataset_name] = sub_samples
        return samples

    def __getitem__(self, index):
        if self.mode == "test":
            sample = self.samples[self.test_on][index]
        else:
            sample = self.samples[index]

        with laspy.open(sample["sub_pc"]) as lidar_file:
            pc = lidar_file.read()

        if sample["dataset"] == "neontree_point_cloud":
            # Only point location are considered for the moment
            point_cloud = np.vstack([pc.x, pc.y, pc.z]).T
        elif sample["dataset"] == "forinstance":
            point_cloud = np.stack([np.array(pc.x), np.array(pc.y), np.array(pc.z)]).T
        else:
            raise Exception("Dataset {} not supported yet")

        if self.seg_task == "semantic_segmentation":
            if sample["dataset"] == "neontree_point_cloud":
                labels = pc.instance_id.copy()
                labels[labels != 0] = 7  # 0-6 are FORinstance
                labels = np.array(labels, dtype=np.int64)
            else:
                labels = np.array(pc.classification, dtype=np.int64)
        else:
            raise Exception("Task {} is not supported yet.".format(self.seg_task))

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
        # normalize coords
        data = self.normalization(data)
        if self.configs["augment"] and self.mode == "train":
            #  Augment PC if required
            data = self.augmentations(data)

        return data, None  # forced by collate

    def collate_fn(self, batch):
        return tuple(zip(*batch))
