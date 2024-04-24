import glob
import os
import random
from pathlib import Path

import cv2 as cv
import einops
import geopandas as gpd
import kornia
import matplotlib.pyplot as plt
import numpy as np
import pyjson5 as json
import rasterio
import rasterio.mask
import torch
from rasterio import features
from rasterio.windows import Window
from tqdm import tqdm
from torchvision import transforms
import albumentations as A
import utilities

"""
Data loading for the Waititu dataset published in:
Kattenborn, Teja, et al. "Convolutional Neural Networks accurately predict cover fractions of plant species and communities in Unmanned Aerial Vehicle imagery."
"""


class WaitituDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        print("=" * 40)
        print("Initializing Waititu dataset - mode: ", mode)
        print("=" * 40)
        self.configs = configs
        self.root_path = os.path.join(configs["root_path"], "Waititu", "data_treespecies_waitutu_nz")
        self.mode = mode
        self.samples = []
        self.full_samples = []
        self.test_captions = ["plot_wcm"]
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None

        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])

        areas = os.listdir(self.root_path)
        for area in areas:
            area_path = os.path.join(self.root_path, area)
            tif_file = glob.glob(os.path.join(area_path, "*.tif"))[0]
            aoi_path = glob.glob(os.path.join(area_path, "*_aoi.shp"))[0]
            metumb_path = glob.glob(os.path.join(area_path, "*metumb.shp"))[0]
            daccup_path = glob.glob(os.path.join(area_path, "*daccup.shp"))[0]
            shape = {"metumb": metumb_path, "daccup": daccup_path}
            tiles_path = os.path.join(area_path, "tiles_" + str(configs["tile_size"]))
            masks_path = os.path.join(area_path, "masks_" + str(configs["tile_size"]))
            sample = {"image": tif_file, "aoi_path": aoi_path, "shapefile": shape}
            self.full_samples.append(sample)
            if area in self.test_captions and self.mode != "test":
                continue
            if self.mode == "test" and area not in self.test_captions:
                continue
            if os.path.exists(tiles_path) and os.path.exists(masks_path) and configs["tiling"] == False:
                print("Tiles for area: ", area)
                print("Skipping tiling.")
                tiles = os.listdir(tiles_path)
                for tile in tiles:
                    id = tile.split("_")[1:]
                    id = "_".join(id)
                    sample = {
                        "image": os.path.join(tiles_path, "tile_" + id),
                        "mask": os.path.join(masks_path, "tile_" + id),
                    }
                    self.samples.append(sample)
            else:
                Path(tiles_path).mkdir(parents=True, exist_ok=True)
                Path(masks_path).mkdir(parents=True, exist_ok=True)
                self.samples.extend(self.tilerize(tif_file, shape, aoi_path, tiles_path, masks_path))

        random.Random(999).shuffle(self.samples)
        if mode == "train":
            self.samples = self.samples[: int(0.9 * len(self.samples))]
        elif mode == "val":
            self.samples = self.samples[int(0.9 * len(self.samples)) :]
        elif mode != "test":
            print("Mode: ", mode, " not supported!")
            exit(2)

        self.num_examples = len(self.samples)
        print("Mode: ", self.mode, " Number of examples: ", self.num_examples)

    def __len__(self):
        return self.num_examples

    def plot(self, index=0):
        sample = self.full_samples[index]
        aoi_path = sample["aoi_path"]
        shape = sample["shapefile"]
        tif_file = sample["image"]
        aoi_shape = gpd.read_file(aoi_path)
        m_shapefile = gpd.read_file(shape["metumb"])
        d_shapefile = gpd.read_file(shape["daccup"])
        with rasterio.open(tif_file) as src:
            metadata = src.profile

            image, out_transform = rasterio.mask.mask(src, aoi_shape.geometry, crop=True)
            metadata["transform"] = out_transform
            metadata["height"] = image.shape[1]
            metadata["width"] = image.shape[2]

            image = einops.rearrange(image, "c h w -> h w c")

            rasterized = np.zeros((metadata["height"], metadata["width"]), dtype=metadata["dtype"])
            m_shapes = tuple((geom, 1) for geom in m_shapefile.geometry)
            d_shapes = tuple((geom, 2) for geom in d_shapefile.geometry)
            shapes = m_shapes + d_shapes
            mask = features.rasterize(shapes=shapes, fill=0, out=rasterized, transform=metadata["transform"])

        _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        ax[0].imshow(mask)
        ax[0].set_title("Mask")

        ax[1].imshow(image[:, :, :3] / image[:, :, :3].max())
        ax[1].set_title(tif_file.split("/")[-2])
        plt.savefig("Waititu_sample" + str(index) + ".png")
        plt.show()

    def tilerize(self, tif_path, shape_path, aoi_path, tiles_path, masks_path):
        print("Initialize tile creation with shapefile paths:")
        print("Mask shapefile: ", shape_path)
        print("AOI shapefile: ", aoi_path)

        metumb_path = shape_path["metumb"]
        daccup_path = shape_path["daccup"]
        metumb_shapefile = gpd.read_file(metumb_path)
        daccup_shapefile = gpd.read_file(daccup_path)
        aoi_shape = gpd.read_file(aoi_path)

        # Crop to AOI
        with rasterio.open(tif_path) as src:
            metadata = src.profile

            tif, out_transform = rasterio.mask.mask(src, aoi_shape.geometry, crop=True)
            metadata["transform"] = out_transform
            metadata["height"] = tif.shape[1]
            metadata["width"] = tif.shape[2]

            rasterized = np.zeros((metadata["height"], metadata["width"]), dtype=metadata["dtype"])
            metumb_shapes = tuple(((geom, 1) for geom in metumb_shapefile.geometry))
            daccub_shapes = tuple(((geom, 2) for geom in daccup_shapefile.geometry))
            shapes = metumb_shapes + daccub_shapes
            mask = features.rasterize(shapes=shapes, fill=0, out=rasterized, transform=metadata["transform"])

        num_rows = tif.shape[1]
        num_cols = tif.shape[2]
        tile_size = self.configs["tile_size"]
        samples = []
        print("Full area size: ", mask.shape)
        print("Desired tile size: ", tile_size)

        print("Saving tiles")

        for row in tqdm(range(0, num_rows, tile_size)):
            for col in tqdm(range(0, num_cols, tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)

                tile = tif[
                    :, window.row_off : window.row_off + window.height, window.col_off : window.col_off + window.width
                ]
                tile_profile = metadata.copy()
                tile_profile.update(
                    {
                        "height": tile.shape[1],
                        "width": tile.shape[2],
                        "transform": rasterio.windows.transform(window, metadata["transform"]),
                    }
                )

                mask_tile = mask[
                    window.row_off : window.row_off + window.height, window.col_off : window.col_off + window.width
                ]
                mask_tile_profile = metadata.copy()

                mask_tile_profile.update(
                    {
                        "height": mask_tile.shape[0],
                        "width": mask_tile.shape[1],
                        "transform": rasterio.windows.transform(window, metadata["transform"]),
                    }
                )

                with rasterio.open(os.path.join(tiles_path, f"tile_{row}_{col}.tif"), "w", **tile_profile) as dst:
                    dst.write(tile)

                mask_tile_profile["count"] = 1
                with rasterio.open(
                    os.path.join(masks_path, f"tile_{row}_{col}.tif"), "w", **mask_tile_profile
                ) as mask_dst:
                    mask_dst.write(mask_tile, indexes=1)
                sample = {
                    "image": os.path.join(tiles_path, f"tile_{row}_{col}.tif"),
                    "mask": os.path.join(masks_path, f"tile_{row}_{col}.tif"),
                }
                samples.append(sample)
        return samples

    # TODO define preprocess pipeline e.g Normalization etc.
    def __getitem__(self, index):
        sample = self.samples[index]
        mask_path = sample["mask"]
        image_path = sample["image"]
        with rasterio.open(image_path) as src:
            image = src.read()

        with rasterio.open(mask_path) as src2:
            mask = src2.read()

        image = image.astype("float32")
        if np.isinf(image).any():
            print("Image is inf: ")
            print(image)
            exit(3)

        if mask.shape[0] != self.configs["tile_size"] or mask.shape[1] != self.configs["tile_size"]:
            image = einops.rearrange(image, "c h w -> h w c")
            resize = A.Compose(
                [A.augmentations.Resize(height=self.configs["tile_size"], width=self.configs["tile_size"], p=1.0)]
            )

            transform = resize(image=image, mask=mask.squeeze())
            image = transform["image"]
            mask = transform["mask"]
            image = einops.rearrange(image, "h w c -> c h w")

        if not self.configs["webdataset"]:
            if self.configs["augment"] and self.mode == "train":
                image = einops.rearrange(image, "c h w -> h w c")
                transform = self.augmentations(image=image, mask=mask.squeeze())
                image = transform["image"]
                mask = transform["mask"]
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

        mask = torch.from_numpy(mask).long()

        return image, mask
