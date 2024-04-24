import os
import pprint
import random

import cv2 as cv
import einops
import kornia
import matplotlib.pyplot as plt
import numpy as np
import pyjson5 as json
import rasterio
import torch
from tqdm import tqdm
from torchvision import transforms
import pyjson5 as json
import glob
import utilities
import albumentations as A

"""
Data loading for the FLAIR Dataset published in:
Garioud, Anatol, et al. "Challenge FLAIR #2: textural and temporal information for semantic segmentation
from multi-source optical imagery."

Test data are not released yet.

"""


class FLAIR2Dataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        self.configs = configs
        self.root_path = os.path.join(configs["root_path"], "FLAIR_2")
        self.mode = mode
        self.labels_path = os.path.join(self.root_path, "flair_labels_train")
        self.root_aerial_path = os.path.join(self.root_path, "flair_aerial_train")
        self.root_sentinel_path = os.path.join(self.root_path, "flair_sen_train")
        self.centroids_to_patch_path = os.path.join(self.root_path, "flair-2_centroids_sp_to_patch.json")
        self.centroids_to_patch = json.load(open(self.centroids_to_patch_path, "r"))
        self.sat_patch_size = configs["sentinel_size"]
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None

        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])
        areas = os.listdir(self.root_aerial_path)

        if self.mode == "train" or self.mode == "val":
            areas = areas[: int(0.9 * len(areas))]
        elif self.mode == "test":
            areas = areas[int(0.9 * len(areas)) :]
        else:
            print(mode, "is not a valid mode! Exiting!")
            exit(2)

        print("=" * 40)
        print("Initializing FLAIR-2 dataset - mode: ", mode)
        print("=" * 40)
        self.samples = []
        for area in tqdm(areas):
            area_aerial_path = os.path.join(self.root_aerial_path, area)
            sub_areas = os.listdir(area_aerial_path)
            for sub_area in sub_areas:
                sub_area_aerial_path = os.path.join(area_aerial_path, sub_area)
                images_list = os.listdir(os.path.join(sub_area_aerial_path, "img"))
                for image in images_list:
                    if image.endswith("tif"):
                        sample = {}
                        sample["aerial"] = os.path.join(sub_area_aerial_path, "img", image)
                        sample["sentinel2"] = os.path.join(self.root_sentinel_path, area, sub_area, "sen")
                        mask_file = sample["aerial"].replace(self.root_aerial_path, self.labels_path)
                        mask_file = mask_file.replace("img", "msk").replace("IMG", "MSK")
                        sample["label"] = mask_file
                        sample["area"] = area
                        sample["sub_area"] = sub_area
                        self.samples.append(sample)

        self.num_examples = len(self.samples)
        print("Mode: ", mode, " - Number of examples :", self.num_examples)

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return torch.from_numpy(array.astype(float))

    def read_superarea_and_crop(self, numpy_file: str, idx_centroid: list) -> np.ndarray:
        data = np.load(numpy_file, mmap_mode="r")
        subset_sp = data[
            :,
            :,
            idx_centroid[0] - int(self.sat_patch_size / 2) : idx_centroid[0] + int(self.sat_patch_size / 2),
            idx_centroid[1] - int(self.sat_patch_size / 2) : idx_centroid[1] + int(self.sat_patch_size / 2),
        ]
        return torch.from_numpy(subset_sp.astype(float))

    def read_labels(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_label:
            labels = src_label.read()[0]
            labels[labels > self.configs["num_classes"]] = self.configs["num_classes"]
            labels = labels - 1
            return labels

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.read_img(sample["aerial"])
        mask_path = sample["label"]
        mask = self.read_labels(mask_path)
        aerial_id = sample["aerial"].split("/")[-1]
        data_path = glob.glob(os.path.join(sample["sentinel2"], "*data.npy"))[0]
        sentinel = self.read_superarea_and_crop(data_path, self.centroids_to_patch[aerial_id])

        if len(self.configs["data_source"]) == 1:
            if self.configs["data_source"] == "sentinel2":
                image = sentinel
                if self.configs["timeseries"]:
                    image = einops.rearrange(image, "t c h w -> (t*c) h w")
                else:
                    # Randomly pick a sentinel image to use
                    choice = random.randint(0, sentinel.shape[0] - 1)
                    image = image[choice]
        elif len(self.configs["data_source"]) == 2:
            if self.configs["timeseries"]:
                # Randomly pick a sentinel image to use
                timeseries_sequence = list(range(sentinel.shape[0]))
                timeseries_subset = sorted(random.sample(timeseries_sequence, self.configs["length_of_sequence"]))
                sentinel = sentinel[timeseries_subset, :, :, :]
                sentinel = einops.rearrange(sentinel, "t c h w -> (t*c) h w")
            else:
                # Randomly pick a sentinel image to use
                choice = random.randint(0, sentinel.shape[0] - 1)
                sentinel = sentinel[choice]
            sentinel = einops.rearrange(sentinel, "c h w -> h w c")
            resize = A.Compose([A.augmentations.Resize(height=image.shape[1], width=image.shape[2], p=1.0)])
            transform = resize(image=sentinel.numpy())
            sentinel = torch.from_numpy(transform["image"])
            sentinel = einops.rearrange(sentinel, "h w c->c h w")

            image = torch.cat((image, sentinel), dim=0)
        else:
            print("FLAIR2 supports only 2 data sources! Validate config file!")
            exit(3)

        image = image.numpy()
        if not self.configs["webdataset"]:
            if self.configs["augment"] and self.mode == "train":
                image = einops.rearrange(image, "c h w -> h w c")
                transform = self.augmentations(image=image, mask=mask)
                image = transform["image"]
                mask = transform["mask"]
                image = einops.rearrange(image, "h w c -> c h w")
            if self.configs["normalization"] == "minmax":
                image /= image.max()
            elif self.configs["normalization"] == "standard":
                image = torch.from_numpy(image).float()
                image = self.normalization(image)
        else:
            image = torch.from_numpy(image).float()

        # Bring labels in range [0,..]
        mask = torch.from_numpy(mask).long()

        return image, mask
