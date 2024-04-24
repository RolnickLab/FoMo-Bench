import os
import pickle
import pprint
import random
import warnings

import cv2 as cv
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjson5 as json
import rasterio
import torch
from tqdm import tqdm

import utilities.augmentations
import albumentations as A

warnings.simplefilter("ignore")
"""
Data loading for the ForestNet Dataset published in:
Irvin, Jeremy, et al. "Forestnet: Classifying drivers of deforestation in indonesia using deep learning on satellite imagery."
"""


class ForestNetDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        self.configs = configs
        self.root_path = os.path.join(configs["root_path"], "ForestNet")
        self.mode = mode
        self.label_category = {"Plantation": 1, "Smallholder agriculture": 2, "Grassland shrubland": 3, "Other": 4}
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None
        record_path = os.path.join(self.root_path, self.mode + ".csv")

        self.metadata = pd.read_csv(record_path)
        self.samples = []
        for index, row in self.metadata.iterrows():
            sample = {}

            sample["label"] = self.label_category[row["merged_label"]]
            sample["year"] = row["year"]
            sample["path"] = os.path.join(self.root_path, row["example_path"])
            sample["forest_loss_path"] = os.path.join(sample["path"], "forest_loss_region.pkl")
            sample["auxiliary_path"] = os.path.join(sample["path"], "auxiliary")
            sample["images_path"] = os.path.join(sample["path"], "images")
            self.samples.append(sample)

        self.num_examples = len(self.samples)

    def __len__(self):
        return self.num_examples

    def plot(self, index=0):
        sample = self.samples[index]
        forest_loss = pickle.load(open(sample["forest_loss_path"], "rb"))

        image = cv.imread(os.path.join(sample["images_path"], "visible", "composite.png"))
        mask = rasterio.features.rasterize([forest_loss], fill=0, out_shape=image.shape[:2])
        mask[mask > 0] = sample["label"]
        infrared = np.load(os.path.join(sample["images_path"], "infrared", "composite.npy"))
        _, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        ax[0].imshow(mask)
        ax[0].set_title("Forest loss mask")
        ax[1].imshow(image)
        ax[1].set_title("Visible satellite image")
        ax[2].imshow(infrared)
        ax[2].set_title("Infrared")
        plt.savefig("ForestNetSample.png")
        plt.show()

    def __getitem__(self, index):
        sample = self.samples[index]
        forest_loss = pickle.load(open(sample["forest_loss_path"], "rb"))
        image = cv.imread(os.path.join(sample["images_path"], "visible", "composite.png")).astype(float)
        mask = rasterio.features.rasterize([forest_loss], fill=0, out_shape=image.shape[:2])

        # Make it a multiclass problem
        mask[mask > 0] = sample["label"]

        # Composite image. According to the paper: A composite image is constructed by taking a per-pixel median over these cloud-filtered scenes, using the five least
        # cloudy scenes when less than five such scenes were available.
        # We don't use it
        composite = np.load(os.path.join(sample["images_path"], "infrared", "composite.npy"))

        # Auxiliary data
        # aux = np.load(os.path.join(sample['images_path'],'auxiliary',..))

        if self.configs["resize"] is not None:
            resize = A.Compose(
                [A.augmentations.Resize(height=self.configs["resize"], width=self.configs["resize"], p=1.0)]
            )
            transform = resize(image=image, mask=mask)
            image = transform["image"]
            mask = transform["mask"]

        label = sample["label"]
        if not self.configs["webdataset"]:
            if self.configs["augment"] and self.mode == "train":
                transform = self.augmentations(image=image, mask=mask)
                image = transform["image"]
                mask = transform["mask"]
            if self.configs["normalization"] == "minmax":
                image /= image.max()
            elif self.configs["normalization"] == "standard":
                image = torch.from_numpy(image).float()
                image = self.normalization(image)
        else:
            image = torch.from_numpy(image).float()

        image = einops.rearrange(image, "h w c -> c h w")
        image = torch.from_numpy(image).float()

        if self.configs["task"] == "segmentation":
            return image, torch.from_numpy(mask).long()
        elif self.configs["task"] == "classification":
            return image, torch.tensor(label).long()
