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

import utilities

"""
Data loading for the FLAIR Dataset published in:
Garioud, Anatol, et al. "FLAIR: French Land cover from Aerospace ImageRy."
"""


class FLAIRDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        self.configs = configs
        self.root_path = os.path.join(configs["root_path"], "FLAIR")
        self.mode = mode
        if self.mode == "train" or self.mode == "val":
            self.labels_path = os.path.join(self.root_path, "flair_labels_train")
            self.root_path = os.path.join(self.root_path, "flair_aerial_train")
        elif self.mode == "test":
            self.labels_path = os.path.join(self.root_path, "flair_1_labels_test")
            self.root_path = os.path.join(self.root_path, "flair_1_aerial_test")

        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None

        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])
        areas = os.listdir(self.root_path)
        print("=" * 40)
        print("Initializing FLAIR dataset - mode: ", mode)
        print("=" * 40)
        self.samples = []
        for area in tqdm(areas):
            area_path = os.path.join(self.root_path, area)
            sub_areas = os.listdir(area_path)
            for sub_area in sub_areas:
                sub_area_path = os.path.join(area_path, sub_area)
                images_list = os.listdir(os.path.join(sub_area_path, "img"))
                for image in images_list:
                    if image.endswith("tif"):
                        sample = {}
                        sample["path"] = os.path.join(sub_area_path, "img", image)
                        mask_file = image.replace("IMG", "MSK")
                        sample["label"] = os.path.join(self.labels_path, area, sub_area, "msk", mask_file)
                        sample["area"] = area
                        sample["sub_area"] = sub_area
                        self.samples.append(sample)
        if mode == "train":
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[: int(0.9 * len(self.samples))]
        elif mode == "val":
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[int(0.9 * len(self.samples)) :]
        self.num_examples = len(self.samples)

    def __len__(self):
        return self.num_examples

    def plot(self, index=0):
        # TOY preprocessing for dev purposes
        sample = self.samples[index]
        path = sample["path"]
        mask_path = sample["label"]
        with rasterio.open(path) as src:
            image = src.read()

        with rasterio.open(mask_path) as src2:
            mask = src2.read()

        image = image[:3]
        image = einops.rearrange(image, "c h w -> h w c")
        _, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].imshow(mask)
        ax[0].set_title("Mask")
        ax[1].imshow(image / image.max())
        ax[1].set_title("Satellite image")
        plt.savefig("sample_" + str(index) + ".png")
        plt.show()

    def __getitem__(self, index):
        # TOY preprocessing for dev purposes
        sample = self.samples[index]
        path = sample["path"]
        mask_path = sample["label"]
        with rasterio.open(path) as src:
            image = src.read()

        with rasterio.open(mask_path) as src2:
            mask = src2.read()

        if not self.configs["webdataset"]:
            if self.configs["augment"] and self.mode == "train":
                transform = self.augmentations(image=image, mask=mask)
                image = transform["image"]
                mask = transform["mask"]
            image = torch.from_numpy(image).float()
            if self.configs["normalization"] == "minmax":
                image /= image.max()
            elif self.configs["normalization"] == "standard":
                image = self.normalization(image)
        else:
            image = torch.from_numpy(image).float()

        # Bring labels in range [0,..]
        mask = torch.from_numpy(mask).long().squeeze() - 1
        return image, mask
