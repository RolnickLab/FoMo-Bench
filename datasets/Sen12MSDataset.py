import glob
import os
import pickle
import random

import albumentations as A
import cv2 as cv
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjson5 as json
import rasterio
import torch
import tqdm
from torchvision import transforms


class Sen12MSDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        print("=" * 40)
        print("Initializing SEN12MS mode - ", mode)
        print("=" * 40)

        self.configs = configs
        self.root_path = os.path.join(configs["root_path"], "Sen12MS")
        self.mode = mode
        if self.mode == "train" or self.mode == "val":
            sample_file = "train_list.txt"
        elif self.mode == "test":
            sample_file = "test_list.txt"
        else:
            print("Uknown phase")
            exit(2)
        self.valid_samples = open(os.path.join(self.root_path, sample_file), "r").readlines()
        self.labels = pickle.load(open(os.path.join(self.root_path, "IGBP_probability_labels.pkl"), "rb"))

        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])

        rois_path = os.path.join(self.root_path, "ROIs")
        self.rois = os.listdir()
        self.samples = []
        for file in self.valid_samples:
            file = file.strip()
            roi = "_".join(file.split("_")[:2])

            s2_folder = "_".join(file.split("_")[2:4])
            s1_folder = s2_folder.replace("s2", "s1")
            lc_folder = s2_folder.replace("s2", "lc")

            s1_path = os.path.join(rois_path, roi, s1_folder, file.replace("_s2_", "_s1_"))
            s2_path = os.path.join(rois_path, roi, s2_folder, file)
            lc_path = os.path.join(rois_path, roi, lc_folder, file.replace("_s2_", "_lc_"))

            label = self.labels[file]
            sample = {"lc_path": lc_path, "s1_path": s1_path, "s2_path": s2_path, "labels": label}
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
        sample = self.samples[index]
        s1_path = sample["s1_path"]
        s2_path = sample["s2_path"]
        with rasterio.open(s1_path) as srcs1:
            s1_patch = srcs1.read()
        with rasterio.open(s2_path) as srcs2:
            s2_patch = srcs2.read()
        labels = sample["labels"]
        labels[labels >= 0.5] = 1
        labels[labels < 0.5] = 0
        _, ax = plt.subplots(nrows=1, ncols=3, figsize=((12, 4)))
        ax[0].imshow(s1_patch[0])
        ax[0].set_title("VV")
        ax[1].imshow(s1_patch[1])
        ax[1].set_title("VH")
        s2_patch = einops.rearrange(s2_patch[1:4, :, :], "c h w -> h w c")
        s2_patch = cv.cvtColor(s2_patch, cv.COLOR_BGR2RGB)
        ax[2].imshow(s2_patch / s2_patch.max())
        ax[2].set_title("RGB")

        plt.savefig("Sen12MS_sample_" + str(index) + ".png")
        plt.show()

    def __getitem__(self, index):
        sample = self.samples[index]
        s1_path = sample["s1_path"]
        s2_path = sample["s2_path"]
        with rasterio.open(s1_path) as srcs1:
            s1_patch = srcs1.read()
        with rasterio.open(s2_path) as srcs2:
            s2_patch = srcs2.read()
        labels = sample["labels"]
        labels[labels >= 0.5] = 1
        labels[labels < 0.5] = 0
        s1_patch = torch.from_numpy(s1_patch.astype("float")).float()
        s2_patch = torch.from_numpy(s2_patch.astype("float")).float()

        s2_patch = torch.clamp(s2_patch, min=0, max=10000)
        s1_patch = torch.clamp(s1_patch, min=-25, max=0)

        # Stack data
        image = torch.cat((s1_patch, s2_patch), dim=0).numpy()

        if not self.configs["webdataset"]:
            if self.configs["augment"] and self.mode == "train":
                image = einops.rearrange(image, "c h w -> h w c")
                transform = self.augmentations(image=image)
                image = transform["image"]
                image = einops.rearrange(image, "h w c -> c h w")
            if self.configs["normalization"] == "minmax":
                s1 = torch.from_numpy(image[:2, :, :]) / 25 + 1
                s2 = torch.from_numpy(image[2:, :, :]) / 10000
                image = torch.cat((s1, s2), dim=0)
            elif self.configs["normalization"] == "standard":
                image = torch.from_numpy(image).float()
                image = self.normalization(image)
            else:
                image = torch.from_numpy(image).float()
        else:
            image = torch.from_numpy(image).float()

        return image, torch.from_numpy(labels)
