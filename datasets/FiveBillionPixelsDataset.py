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
import rioxarray as rio
import tqdm
import utilities
from pathlib import Path
import random


class FiveBillionPixelsDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        print("=" * 40)
        print("Initializing FiveBillionPixels Dataset mode - ", mode)
        print("=" * 40)
        self.mode = mode
        self.configs = configs
        self.root_path = os.path.join(configs["root_path"], "FiveBillionPixels")
        self.image_path = os.path.join(self.root_path, "Image_16bit_BGRNir")
        self.label_path = os.path.join(self.root_path, "Annotation__index")
        self.tile_path = os.path.join(self.root_path, "tiles")
        images = os.listdir(self.image_path)
        self.samples = []
        self.tile_path = os.path.join(self.root_path, "tiles", mode)
        self.tile_label_path = os.path.join(self.root_path, "tile_labels", mode)
        self.val_scenes = [
            "GF2_PMS1__L1A0001094941-MSS1.tiff",
            "GF2_PMS1__L1A0001680853-MSS1.tiff",
            "GF2_PMS2__L1A0000958144-MSS2.tiff",
            "GF2_PMS2__L1A0001757317-MSS2.tiff",
            "GF2_PMS1__L1A0001491417-MSS1.tiff",
            "GF2_PMS2__L1A0000564691-MSS2.tiff",
            "GF2_PMS2__L1A0001206072-MSS2.tiff",
            "GF2_PMS2__L1A0001573999-MSS2.tiff",
            "GF2_PMS2__L1A0001886305-MSS2.tiff",
        ]
        self.test_scenes = [
            "GF2_PMS1__L1A0001064454-MSS1.tiff",
            "GF2_PMS1__L1A0001118839-MSS1.tiff",
            "GF2_PMS1__L1A0001344822-MSS1.tiff",
            "GF2_PMS1__L1A0001348919-MSS1.tiff",
            "GF2_PMS1__L1A0001366278-MSS1.tiff",
            "GF2_PMS1__L1A0001366284-MSS1.tiff",
            "GF2_PMS1__L1A0001395956-MSS1.tiff",
            "GF2_PMS1__L1A0001432972-MSS1.tiff",
            "GF2_PMS1__L1A0001670888-MSS1.tiff",
            "GF2_PMS1__L1A0001680857-MSS1.tiff",
            "GF2_PMS1__L1A0001680858-MSS1.tiff",
            "GF2_PMS1__L1A0001757429-MSS1.tiff",
            "GF2_PMS1__L1A0001765574-MSS1.tiff",
            "GF2_PMS2__L1A0000607677-MSS2.tiff",
            "GF2_PMS2__L1A0000607681-MSS2.tiff",
            "GF2_PMS2__L1A0000718813-MSS2.tiff",
            "GF2_PMS2__L1A0001038935-MSS2.tiff",
            "GF2_PMS2__L1A0001038936-MSS2.tiff",
            "GF2_PMS2__L1A0001119060-MSS2.tiff",
            "GF2_PMS2__L1A0001367840-MSS2.tiff",
            "GF2_PMS2__L1A0001378491-MSS2.tiff",
            "GF2_PMS2__L1A0001378501-MSS2.tiff",
            "GF2_PMS2__L1A0001396036-MSS2.tiff",
            "GF2_PMS2__L1A0001396037-MSS2.tiff",
            "GF2_PMS2__L1A0001416129-MSS2.tiff",
            "GF2_PMS2__L1A0001471436-MSS2.tiff",
            "GF2_PMS2__L1A0001517494-MSS2.tiff",
            "GF2_PMS2__L1A0001591676-MSS2.tiff",
            "GF2_PMS2__L1A0001787564-MSS2.tiff",
        ]

        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None
        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])

        if not self.configs["tilerize"] or not os.path.isdir(self.tile_path):
            for index, image in tqdm.tqdm(enumerate(images)):
                if ".DS_Store" in image:
                    continue
                image_path = os.path.join(self.image_path, image)
                image_name = image.split(".")[0]

                if image in self.val_scenes and mode != "val":
                    continue
                if image in self.test_scenes and mode != "test":
                    continue
                if image not in self.val_scenes and mode == "val":
                    continue
                if image not in self.test_scenes and mode == "test":
                    continue
                label_path = os.path.join(self.label_path, image_name + "_24label.png")
                if not self.configs["tilerize"]:
                    self.samples.append({"image": image_path, "label": label_path})
                else:
                    if not os.path.isdir(self.tile_path):
                        path = Path(self.tile_path)
                        path.mkdir(parents=True, exist_ok=True)
                    if not os.path.isdir(self.tile_label_path):
                        path = Path(self.tile_label_path)
                        path.mkdir(parents=True, exist_ok=True)
                    self.samples.extend(self.tilerize(image_path, label_path, image_name))
        else:
            images = os.listdir(self.tile_path)
            self.samples = []
            for index, image in tqdm.tqdm(enumerate(images)):
                image_path = os.path.join(self.tile_path, image)
                image_name = image.split(".")[0]
                label_path = os.path.join(self.tile_label_path, image_name + ".png")
                self.samples.append({"image": image_path, "label": label_path})

        random.Random(999).shuffle(self.samples)
        print(self.samples[0])
        print("Samples for mode: ", mode, " = ", len(self.samples))
        self.num_examples = len(self.samples)

    def tilerize(self, image_path, label_path, image_name):
        tif = rio.open_rasterio(image_path, engine="rasterio").sel(band=[3, 2, 1, 4])

        image = tif.to_numpy()
        label = cv.imread(label_path, 0)
        image = einops.rearrange(image, "c h w -> h w c")
        label = label[: image.shape[0], : image.shape[1]]

        tiles = []
        for i in tqdm.tqdm(range(0, image.shape[0], self.configs["tile_size"])):
            for j in range(0, image.shape[1], self.configs["tile_size"]):
                pad_x = False
                pad_y = False
                if image.shape[1] <= j + self.configs["tile_size"]:
                    xmax_step = image.shape[1] - j
                    pad_x = True
                else:
                    xmax_step = self.configs["tile_size"]

                if image.shape[0] <= i + self.configs["tile_size"]:
                    ymax_step = image.shape[0] - i
                    pad_y = True
                else:
                    ymax_step = self.configs["tile_size"]

                transform = A.augmentations.Crop(p=1.0, x_min=j, y_min=i, x_max=j + xmax_step, y_max=i + ymax_step)
                tile_transform = transform(image=image, mask=label)
                tile_image = tile_transform["image"]
                tile_mask = tile_transform["mask"]

                if pad_x or pad_y:
                    tmp_tile = np.zeros((120, 120, 4))
                    tmp_label = np.zeros((120, 120))

                    tmp_tile[0 : tile_image.shape[0], 0 : tile_image.shape[1], :] = tile_image
                    tmp_label[0 : tile_image.shape[0], 0 : tile_image.shape[1]] = tile_mask

                    tile_image = tmp_tile
                    tile_mask = tmp_label

                tile_id = os.path.join(self.tile_path, image_name + "_" + str(i) + "_" + str(j) + ".npy")
                label_id = os.path.join(self.tile_label_path, image_name + "_" + str(i) + "_" + str(j) + ".png")
                tile_image = tile_image.astype(np.float32)
                with open(tile_id, "wb") as f:
                    np.save(f, tile_image)
                cv.imwrite(label_id, tile_mask)
                record = {"image": tile_id, "label": label_id}
                tiles.append(record)

        return tiles

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample["image"]
        label = cv.imread(sample["label"], 0)

        if not self.configs["tilerize"]:
            tif = rio.open_rasterio(image_path, engine="rasterio").sel(band=[3, 2, 1, 4])
            image = tif.to_numpy()
            image = image[:, :6907, :7300]
            label = label[:6907, :7300]
            image = einops.rearrange(image, "c h w -> h w c")
        else:
            with open(image_path, "rb") as f:
                image = np.load(f)
        image = image.astype(np.float32)

        image = einops.rearrange(image, "h w c->c h w")
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
        return image, torch.from_numpy(label)
