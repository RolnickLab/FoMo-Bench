import os

import albumentations as A
import cv2 as cv
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjson5 as json
import rasterio
import torch
from torchvision import transforms

import utilities


class BigEarthNetDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        print("=" * 40)
        print("Initializing BigEarthNet-MM mode - ", mode)
        print("=" * 40)

        self.configs = configs
        self.root_path = os.path.join(configs["root_path"], "BigEarthNet")
        self.mode = mode
        self.s1_path = os.path.join(self.root_path, "BigEarthNet-S1-v1.0")
        self.s2_path = os.path.join(self.root_path, "BigEarthNet-v1.0")
        mappings_path = os.path.join(self.root_path, "s1s2_mapping.csv")
        self.label_indices_path = os.path.join(self.root_path, "label_indices.json")
        self.mappings = pd.read_csv(mappings_path, header=None)
        print("Shuffling mappings")
        self.mappings = self.mappings.sample(frac=1).reset_index(drop=True)
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None
        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])

        # radar and spectral band names to read related GeoTIFF files
        self.band_names_s1 = ["VV", "VH"]
        self.band_names_s2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

        self.splits_path = os.path.join(self.root_path, mode + ".csv")
        self.valid_ids = pd.read_csv(self.splits_path, header=None)
        self.mappings = pd.merge(self.mappings, self.valid_ids, how="inner")
        print(mode, " samples: ", len(self.mappings))

        self.total_label = json.load(open(self.label_indices_path, "r"))
        self.label_indices = self.total_label["original_labels"]
        self.conversion = self.total_label["label_conversion"]

        self.num_examples = len(self.mappings)

    def __len__(self):
        return self.num_examples

    def prepare_array(self, patch_dict, resize=120):
        patch = None
        for key in patch_dict.keys():
            if patch is None:
                patch = patch_dict[key]
                patch = cv.resize(patch, (resize, resize))
                patch = einops.rearrange(patch, "h w -> 1 h w")
            else:
                channel = cv.resize(patch_dict[key], (resize, resize))
                channel = einops.rearrange(channel, "h w -> 1 h w")
                patch = np.vstack((patch, channel))
        return patch

    def read_all_bands(self, path, patch, bands):
        all_bands = {}
        for band in bands:
            band_path = os.path.join(path, patch + "_" + band + ".tif")
            with rasterio.open(band_path) as src:
                band_array = src.read().squeeze()

            all_bands[band] = band_array
        return all_bands

    def prepare_pairs(self, row):
        s2 = self.mappings.iloc[row][0]
        s1 = self.mappings.iloc[row][1]

        s2_patch_dict = self.read_all_bands(os.path.join(self.s2_path, s2), s2, self.band_names_s2)
        s2_patch = self.prepare_array(s2_patch_dict)

        s1_patch_dict = self.read_all_bands(os.path.join(self.s1_path, s1), s1, self.band_names_s1)
        s1_patch = self.prepare_array(s1_patch_dict)

        label_path = os.path.join(self.s2_path, s2, s2 + "_labels_metadata.json")
        file = open(label_path, "r")
        label = json.load(file)["labels"]

        one_hot = np.zeros((43,))
        for i in label:
            one_hot[self.label_indices[i]] = 1

        return (s2_patch, s1_patch), one_hot

    def plot(self, index=0):
        (s2_patch, s1_patch), one_hot = self.prepare_pairs(index)
        inverted_labels = {v: k for k, v in self.total_label["BigEarthNet-19_labels"].items()}
        labels = []
        for idx, elem in enumerate(self.total_label["label_conversion"]):
            for i in elem:
                if one_hot[i] == 1:
                    labels.append(inverted_labels[idx])
        labels = list(np.unique(labels))
        labels = ", ".join(labels)
        _, ax = plt.subplots(nrows=1, ncols=3, figsize=((12, 4)))
        ax[0].imshow(s1_patch[0])
        ax[0].set_title("VV")
        ax[1].imshow(s1_patch[1])
        ax[1].set_title("VH")
        s2_patch = einops.rearrange(s2_patch[:3, :, :], "c h w -> h w c")
        s2_patch = cv.cvtColor(s2_patch, cv.COLOR_BGR2RGB)
        ax[2].imshow(s2_patch / s2_patch.max())
        ax[2].set_title("RGB")
        text = " " * 70 + "Labels:\n" + labels
        plt.text(0.25, 0.9, text, fontsize=8, transform=plt.gcf().transFigure)
        plt.savefig("BigEarthNet_sample" + str(index) + ".png")
        plt.show()

    def __getitem__(self, index):
        (s2_patch, s1_patch), one_hot = self.prepare_pairs(index)

        s2_patch = torch.from_numpy(s2_patch.astype("float"))
        s1_patch = torch.from_numpy(s1_patch.astype("float"))

        image = torch.cat((s1_patch, s2_patch), dim=0).float().numpy()

        labels_19 = np.zeros((19,))
        for idx, elem in enumerate(self.total_label["label_conversion"]):
            for i in elem:
                if one_hot[i] == 1:
                    labels_19[idx] = 1

        if not self.configs["webdataset"]:
            if self.configs["augment"] and self.mode == "train":
                image = einops.rearrange(image, "c h w -> h w c")
                transform = self.augmentations(image=image)
                image = transform["image"]
                image = einops.rearrange(image, "h w c -> c h w")
            if self.configs["normalization"] == "minmax":
                image /= image.max()
            elif self.configs["normalization"] == "standard":
                image = torch.from_numpy(image).float()
                image = self.normalization(image)
            else:
                image = torch.from_numpy(image).float()
        else:
            image = torch.from_numpy(image).float()

        return image, torch.from_numpy(labels_19)
