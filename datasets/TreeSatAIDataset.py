import glob
import os
import random

import cv2 as cv
import einops
import matplotlib.pyplot as plt
import numpy as np
import pyjson5 as json
import rasterio
import torch
from torchvision import transforms
from tqdm import tqdm
import albumentations as A

"""
Data loading for TreeSat published in :
S. Ahlswede, C. Schulz, C. Gava, P. Helber, B. Bischke, M. Frster, F. Arias, J. Hees, B. Demir,
and B. Kleinschmit. TreeSatAI Benchmark Archive: A multi-sensor, multi-label dataset for tree
species classification in remote sensing. ESSD, 2022.
"""


class TreeSatAIDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        print("=" * 20)
        print("Initializing TreeSatAI dataset - mode: ", mode)
        print("=" * 20)
        self.configs = configs
        self.root_path = os.path.expandvars(os.path.join(configs["root_path"], "TreeSat"))
        self.mode = mode
        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])
        self.data_sources = []

        for source in self.configs["data_source"]:
            if source == "aerial_60m":
                self.data_sources.append("aerial_60m")
            elif source == "s1_60m":
                self.data_sources.append("s1/60m")
            elif source == "s2_60m":
                self.data_sources.append("s2/60m")
            elif source == "s1_200m":
                self.data_sources.append("s1/200m")
            elif source == "s2_200m":
                self.data_sources.append("s2/200m")
            else:
                print("Data source: ", source, " not supported.")
                exit(2)
        self.species = os.listdir(os.path.join(self.root_path, "aerial_60m"))
        labels_file = open(os.path.join(self.root_path, "labels", "TreeSatBA_v9_60m_multi_labels.json"), "r")
        self.labels = json.load(labels_file)
        all_species = []
        for v in self.labels.values():
            for item in v:
                sp, v = item
                all_species.append(sp)
        all_species = np.unique(all_species)
        print(all_species)
        print("Num species: ", len(all_species))
        self.species_dict = {}
        for idx, sp in enumerate(all_species):
            self.species_dict[sp] = idx
        self.valid_samples = None
        if self.mode == "train" or self.mode == "val":
            self.valid_samples = open(os.path.join(self.root_path, "train_filenames.lst"), "r").readlines()
        elif self.mode == "test":
            self.valid_samples = open(os.path.join(self.root_path, "test_filenames.lst"), "r").readlines()
        else:
            print("Mode not supported")
            exit(2)

        self.valid_samples = [v.strip() for v in self.valid_samples]

        self.samples = []

        for data_source in self.data_sources:
            if data_source == "aerial_60m":
                for plant in tqdm(self.species):
                    sample = {}
                    sample["data"] = {}
                    files = os.listdir(os.path.join(self.root_path, data_source, plant))
                    aerial_file = None
                    for file in files:
                        if file in self.valid_samples:
                            aerial_file = file
                            sample["data"][data_source] = os.path.join(self.root_path, data_source, plant, aerial_file)
                            sample["labels"] = self.labels[aerial_file]
                            self.samples.append(sample)
            elif "s2/60m" in data_source:
                files = os.listdir(os.path.join(self.root_path, data_source))
                for file in files:
                    if file in self.valid_samples:
                        sample = {}
                        sample["data"] = {}
                        sample["data"][data_source] = os.path.join(self.root_path, data_source, file)
                        if file in self.labels:
                            sample["labels"] = self.labels[file]
                        else:
                            continue
                        self.samples.append(sample)
            elif "s2/200m" in data_source:
                files = os.listdir(os.path.join(self.root_path, data_source))
                for file in files:
                    if file in self.valid_samples:
                        sample = {}
                        sample["data"] = {}
                        sample["data"][data_source] = os.path.join(self.root_path, data_source, file)
                        if file in self.labels:
                            sample["labels"] = self.labels[file]
                        else:
                            continue
                        self.samples.append(sample)

        if self.mode == "train" or self.mode == "val":
            random.Random(999).shuffle(self.samples)
            if self.mode == "train":
                self.samples = self.samples[: int(0.8 * len(self.samples))]
            else:
                self.samples = self.samples[int(0.8 * len(self.samples)) :]

        self.num_examples = len(self.samples)
        print("Number of samples: ", self.num_examples)

    def __len__(self):
        return self.num_examples

    def plot(self, index=0):
        sample = self.samples[index]

        data = sample["data"]
        label = sample["labels"]
        num_plots = len(self.data_sources)
        if "s1/60m" in self.data_sources:
            num_plots += 1
        _, ax = plt.subplots(nrows=1, ncols=num_plots, figsize=(12, 4))

        for idx, source in enumerate(self.data_sources):
            with rasterio.open(data[source]) as src:
                if "s1/60m" in self.data_sources[:idx]:
                    idx += 1
                img = src.read()
                img = einops.rearrange(img[:3, :, :], "c h w -> h w c")
                if "s2" in source:
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img, (304, 304))
                elif "s1" in source:
                    img = cv.resize(img, (304, 304))
                    ax[idx].imshow(img[:, :, 0] / img[:, :, 0].max())
                    ax[idx].set_title("VV")
                    ax[idx + 1].imshow(img[:, :, 1] / img[:, :, 1].max())
                    ax[idx + 1].set_title("VH")
                    continue
                ax[idx].imshow(img / img.max())
                ax[idx].set_title(source)

        labels = list(np.unique(label))
        labels = ", ".join(labels)
        text = " " * 40 + "Labels:\n" + labels
        plt.text(0.4, 0.9, text, fontsize=8, transform=plt.gcf().transFigure)
        plt.savefig("TreeSat_sample_" + str(index) + ".png")

    def __getitem__(self, index):
        sample = self.samples[index]

        data = sample["data"]
        label = sample["labels"]
        image = None
        for source in self.data_sources:
            if len(data.keys()) == 0:
                print("Empty!")
                print(data)
                pass
            with rasterio.open(data[source]) as src:
                img = src.read()

                if source == "s2/200m" and (img.shape[1] != 20 or img.shape[2] != 20):
                    img = einops.rearrange(img, "c h w -> h w c")
                    resize = A.Compose([A.augmentations.Resize(height=20, width=20, p=1.0)])
                    transform = resize(image=img)
                    img = transform["image"]
                    img = einops.rearrange(img, "h w c->c h w")
                if image is None:
                    img = img.astype(np.float32)
                    image = torch.from_numpy(img)
                else:
                    image = torch.cat((image, img), dim=0)
        image = image.numpy()
        if self.configs["enforce_resize"] is not None:
            image = einops.rearrange(image, "c h w -> h w c")
            resize = A.Compose(
                [
                    A.augmentations.Resize(
                        height=self.configs["enforce_resize"], width=self.configs["enforce_resize"], p=1.0
                    )
                ]
            )
            transform = resize(image=image)
            image = transform["image"]
            image = einops.rearrange(image, "h w c -> c h w")
        if not self.configs["webdataset"]:
            if self.configs["augment"] and self.mode == "train":
                image = einops.rearrange(image, "c h w -> h w c")
                transform = self.augmentations(image=image)
                image = transform["image"]
                image = einops.rearrange(image, "h w c -> c h w")
            if self.configs["normalization"] == "minmax":
                image /= image.max() + 1e-5
            elif self.configs["normalization"] == "standard":
                image = torch.from_numpy(image).float()
                image = self.normalization(image)
            else:
                image = torch.from_numpy(image).float()
        else:
            image = torch.from_numpy(image).float()

        final_label = torch.zeros(len(self.species_dict))
        for l in label:
            if self.configs["filter_threshold"] is None:
                final_label[self.species_dict[l[0]]] = 1
            else:
                if l[1] > self.configs["filter_threshold"]:
                    final_label[self.species_dict[l[0]]] = 1
        label = final_label.float()

        return image, label
