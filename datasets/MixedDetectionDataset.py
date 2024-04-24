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
from datasets.ReforesTreeDataset import ReforesTreeDataset


class MixedDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train", test_on="neontree_detection"):
        self.configs = configs
        self.mode = mode
        self.test_on = test_on
        self.dataset_names = self.configs["dataset_names"].split(",")
        self.nb_datasets = len(self.dataset_names)
        self.datasets = self._load_datasets()
        self.samples = self._group_samples()
        # This class support only RGB data for the moment
        self.modality = "rgb"

        if "det_format" in self.configs.keys():
            self.det_format = self.configs["det_format"]
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None

        # Full debugging
        # if self.mode in ('train', 'val'):
        # Debugging train/val
        # self.samples = self.samples[:100]
        # else:
        # Debugging test
        # self.samples['neontree_detection'] = self.samples['neontree_detection'][:100]
        # self.samples['reforestree'] = self.samples['reforestree'][:100]

        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])

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
            if dataset_name.lower() == "neontree_detection":
                self.configs["dataset_name"] = dataset_name.lower()
                self.configs["root_path"] = self.configs["root_path_neontree"]
                datasets[dataset_name.lower()] = NeonTreeDataset(self.configs, self.mode)
            if dataset_name.lower() == "reforestree":
                self.configs["dataset_name"] = dataset_name.lower()
                self.configs["root_path"] = self.configs["root_path_reforestree"]
                datasets[dataset_name.lower()] = ReforesTreeDataset(self.configs, self.mode)
            if len(datasets) == 0:
                raise Exception("Dataset {} not supported".format(dataset_name))
        return datasets

    def _group_samples(self):
        if self.mode in ("train", "val"):
            samples = []
            for dataset_name in self.datasets.keys():
                sub_samples = self.datasets[dataset_name].samples
                # Need to add dataset name to each sample
                sub_samples = self._format_labels(dataset_name, sub_samples)
                samples += sub_samples
            random.Random(999).shuffle(samples)
        else:  # Supposed to be test
            samples = dict()
            for dataset_name in self.datasets.keys():
                sub_samples = self.datasets[dataset_name].samples
                sub_samples = self._format_labels(dataset_name, sub_samples)
                # Need to add dataset name to each sample
                samples[dataset_name] = sub_samples
        return samples

    def _format_labels(self, dataset_name, samples):
        attrib_name = "det_format_" + dataset_name
        if self.configs[attrib_name] != self.configs["det_format"]:
            if self.configs[attrib_name] == "coco":
                for sample in samples:
                    # coco -> pascal
                    formatted_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in sample["boxes"]]
                    sample["bboxes"] = formatted_boxes
                    sample["dataset_name"] = dataset_name  # good moment to add this
            elif self.configs[attrib_name] == "pascal_voc":
                for sample in samples:
                    # pascal -> coco
                    formatted_boxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in sample["boxes"]]
                    sample["bboxes"] = formatted_boxes
                    sample["dataset_name"] = dataset_name  # good moment to add this
            else:
                raise Exception("Detection format {} not supported".format(self.configs[attrib_name]))
        else:
            # still need simple formatting
            for sample in samples:
                if "boxes" in sample.keys():
                    sample["bboxes"] = sample["boxes"]
                sample["dataset_name"] = dataset_name
        return samples

    def __getitem__(self, index):
        if self.mode == "test":
            sample = self.samples[self.test_on][index]
        else:
            sample = self.samples[index]
        if self.modality == "rgb":
            with rasterio.open(sample["image"]) as rgb_file:
                image = rgb_file.read()
            bboxes = sample["bboxes"]
            if sample["dataset_name"] == "neontree_detection":
                class_labels = ["tree"] * (len(bboxes))
            else:
                class_labels = sample["categories"]
            if self.det_format == "coco":
                # [x_min, y_min, x_max, y_max] -> [x_min, y_min, w, h]
                bboxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in bboxes]
            elif self.det_format == "yolo":
                # [x_min, y_min, x_max, y_max] -> [x_min, x_max, y_min, y_max] -> [x_center, y_center, w, h] in relative coords
                img_size = image.shape[1:]
                bboxes = [format_bboxes_voc_to_yolo([box[0], box[2], box[1], box[3]], img_size) for box in bboxes]

            if self.configs["augment"] and self.mode == "train":
                image = einops.rearrange(image, "c h w -> h w c")
                transform = self.augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
                image = transform["image"]
                bboxes = transform["bboxes"]
                class_labels = transform["class_labels"]
                image = einops.rearrange(image, "h w c -> c h w")

            if self.mode in ("val", "test") and "Resize" in list(self.configs["augmentations"].keys()):
                image = einops.rearrange(image, "c h w -> h w c")
                size = self.configs["augmentations"]["Resize"]["value"]
                resizer = A.Compose(
                    [Resize(height=size, width=size, p=1.0)],
                    bbox_params=A.BboxParams(format=self.det_format, min_visibility=0.01, label_fields=["class_labels"]),
                )
                transform = resizer(image=image, bboxes=bboxes, class_labels=class_labels)
                image = transform["image"]
                bboxes = transform["bboxes"]
                class_labels = transform["class_labels"]
                image = einops.rearrange(image, "h w c -> c h w")

            image = torch.from_numpy(image).float()
            if self.configs["normalization"] == "minmax":
                image /= image.max()
            elif self.configs["normalization"] == "standard":
                image = self.normalization(image)
            elif self.configs["normalization"] == "none":
                pass
            else:
                image /= 255.0

            bboxes = np.array(bboxes)
            if self.det_format == "coco":
                areas = bboxes[:, 2] * bboxes[:, 3]
            elif self.det_format == "yolo":
                # not required
                pass
            else:
                # pascal voc format
                areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            if sample["dataset_name"] == "neontree_detection":
                labels = torch.zeros(len(bboxes), dtype=torch.int64)
            else:
                labels = torch.tensor(sample["cat_id"], dtype=torch.int64)
            iscrowd = torch.zeros(len(bboxes), dtype=torch.int64)

            target = dict()
            if self.det_format == "coco":
                ann_info = {
                    "id": torch.tensor(index),
                    "image_id": torch.tensor(index),
                    "category_id": labels,
                    "iscrowd": iscrowd,
                    "area": torch.tensor(areas),
                    "bbox": torch.tensor(bboxes).float(),
                    "segmentation": None,
                }
                target["annotations"] = ann_info
                target["image_id"] = torch.tensor(index)
            elif self.det_format == "yolo":
                bboxes = torch.tensor(bboxes)
                target = torch.hstack((labels.unsqueeze(1), bboxes))
            else:
                # pascal voc format
                target = {
                    "image_id": torch.tensor(index),
                    "boxes": torch.tensor(bboxes).float(),
                    "area": torch.tensor(areas),
                    "iscrowd": iscrowd,
                    "labels": labels,
                    "num_boxes": sample["num_boxes"],
                }
            return image, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))
