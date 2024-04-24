# Code is mostly based on https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/ssl4eo.py

import glob
import os
import random
from typing import Callable, Optional, TypedDict

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pyjson5 as json
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import check_integrity, download_url, extract_archive
from tqdm import tqdm
import utilities
import einops
from torchvision import transforms


class SSL4EOL(NonGeoDataset):
    """SSL4EO-L dataset.

    Landsat version of SSL4EO.

    The dataset consists of a parallel corpus (same locations and dates for SR/TOA)
    for the following sensors:

    .. list-table::
       :widths: 10 10 10 10 10
       :header-rows: 1

       * - Satellites
         - Sensors
         - Level
         - # Bands
         - Link
       * - Landsat 4--5
         - TM
         - TOA
         - 7
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_TOA>`__
       * - Landsat 7
         - ETM+
         - SR
         - 6
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2>`__
       * - Landsat 7
         - ETM+
         - TOA
         - 9
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_TOA>`__
       * - Landsat 8--9
         - OLI+TIRS
         - TOA
         - 11
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA>`__
       * - Landsat 8--9
         - OLI
         - SR
         - 7
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2>`__

    Each patch has the following properties:

    * 264 x 264 pixels
    * Resampled to 30 m resolution (7920 x 7920 m)
    * Single multispectral GeoTIFF file

    .. note::

       Each split is 300--400 GB and requires 3x that to concatenate and extract
       tarballs. Tarballs can be safely deleted after extraction to save space.
       The dataset takes about 1.5 hrs to download and checksum and another 3 hrs
       to extract.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2306.09424

    .. versionadded:: 0.5
    """  # noqa: E501

    class _Metadata(TypedDict):
        num_bands: int
        rgb_bands: list[int]

    metadata: dict[str, _Metadata] = {
        "tm_toa": {"num_bands": 7, "rgb_bands": [2, 1, 0]},
        "etm_toa": {"num_bands": 9, "rgb_bands": [2, 1, 0]},
        "etm_sr": {"num_bands": 6, "rgb_bands": [2, 1, 0]},
        "oli_tirs_toa": {"num_bands": 11, "rgb_bands": [3, 2, 1]},
        "oli_sr": {"num_bands": 7, "rgb_bands": [3, 2, 1]},
    }

    url = "https://hf.co/datasets/torchgeo/ssl4eo_l/resolve/main/{0}/ssl4eo_l_{0}.tar.gz{1}"  # noqa: E501
    checksums = {
        "tm_toa": {
            "aa": "553795b8d73aa253445b1e67c5b81f11",
            "ab": "e9e0739b5171b37d16086cb89ab370e8",
            "ac": "6cb27189f6abe500c67343bfcab2432c",
            "ad": "15a885d4f544d0c1849523f689e27402",
            "ae": "35523336bf9f8132f38ff86413dcd6dc",
            "af": "fa1108436034e6222d153586861f663b",
            "ag": "d5c91301c115c00acaf01ceb3b78c0fe",
        },
        "etm_toa": {
            "aa": "587c3efc7d0a0c493dfb36139d91ccdf",
            "ab": "ec34f33face893d2d8fd152496e1df05",
            "ac": "947acc2c6bc3c1d1415ac92bab695380",
            "ad": "e31273dec921e187f5c0dc73af5b6102",
            "ae": "43390a47d138593095e9a6775ae7dc75",
            "af": "082881464ca6dcbaa585f72de1ac14fd",
            "ag": "de2511aaebd640bd5e5404c40d7494cb",
            "ah": "124c5fbcda6871f27524ae59480dabc5",
            "ai": "12b5f94824b7f102df30a63b1139fc57",
        },
        "etm_sr": {
            "aa": "baa36a9b8e42e234bb44ab4046f8f2ac",
            "ab": "9fb0f948c76154caabe086d2d0008fdf",
            "ac": "99a55367178373805d357a096d68e418",
            "ad": "59d53a643b9e28911246d4609744ef25",
            "ae": "7abfcfc57528cb9c619c66ee307a2cc9",
            "af": "bb23cf26cc9fe156e7a68589ec69f43e",
            "ag": "97347e5a81d24c93cf33d99bb46a5b91",
        },
        "oli_tirs_toa": {
            "aa": "4711369b861c856ebfadbc861e928d3a",
            "ab": "660a96cda1caf54df837c4b3c6c703f6",
            "ac": "c9b6a1117916ba318ac3e310447c60dc",
            "ad": "b8502e9e92d4a7765a287d21d7c9146c",
            "ae": "5c11c14cfe45f78de4f6d6faf03f3146",
            "af": "5b0ed3901be1000137ddd3a6d58d5109",
            "ag": "a3b6734f8fe6763dcf311c9464a05d5b",
            "ah": "5e55f92e3238a8ab3e471be041f8111b",
            "ai": "e20617f73d0232a0c0472ce336d4c92f",
        },
        "oli_sr": {
            "aa": "ca338511c9da4dcbfddda28b38ca9e0a",
            "ab": "7f4100aa9791156958dccf1bb2a88ae0",
            "ac": "6b0f18be2b63ba9da194cc7886dbbc01",
            "ad": "57efbcc894d8da8c4975c29437d8b775",
            "ae": "2594a0a856897f3f5a902c830186872d",
            "af": "a03839311a2b3dc17dfb9fb9bc4f9751",
            "ag": "6a329d8fd9fdd591e400ab20f9d11dea",
        },
    }

    def __init__(
        self,
        configs,
        mode="train",
        checksum: bool = False,
    ) -> None:
        """Initialize a new SSL4EOL instance.

        Args:
            root: root directory where dataset can be found
            split: one of ['tm_toa', 'etm_toa', 'etm_sr', 'oli_tirs_toa', 'oli_sr']
            seasons: number of seasonal patches to sample per location, 1--4
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if any arguments are invalid
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        split = configs["split"]
        seasons = configs["seasons"]

        assert split in self.metadata
        assert seasons in range(1, 5)

        self.configs = configs
        root = configs["root_path"]
        self.root = root
        self.subdir = os.path.join(root, f"ssl4eo_l_{split}")
        self.split = split
        self.seasons = seasons
        self.download = configs["download"]
        self.checksum = checksum
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None
        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])

        self._verify()
        if mode != "train":
            self.scenes = []
        else:
            self.scenes = sorted(os.listdir(self.subdir))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image sample
        """
        root = os.path.join(self.subdir, self.scenes[index])
        subdirs = os.listdir(root)
        subdirs = random.sample(subdirs, self.seasons)

        images = []
        for subdir in subdirs:
            directory = os.path.join(root, subdir)
            filename = os.path.join(directory, "all_bands.tif")
            with rasterio.open(filename) as f:
                image = f.read()
                images.append(torch.from_numpy(image.astype(np.float32)).unsqueeze(0))

        # sample = {"image": torch.cat(images)}
        image = torch.cat(images)

        image = image.numpy()
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

        return image, 0

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.scenes)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        path = os.path.join(self.subdir, "00000*", "*", "all_bands.tif")
        if glob.glob(path):
            return

        # Check if the tar.gz files have already been downloaded
        exists = []
        for suffix in self.checksums[self.split]:
            path = self.subdir + f".tar.gz{suffix}"
            exists.append(os.path.exists(path))

        if all(exists):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for suffix, md5 in self.checksums[self.split].items():
            download_url(
                self.url.format(self.split, suffix),
                self.root,
                md5=md5 if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        # Concatenate all tarballs together
        chunk_size = 2**15  # same as torchvision
        path = self.subdir + ".tar.gz"
        with open(path, "wb") as f:
            for suffix in self.checksums[self.split]:
                with open(path + suffix, "rb") as g:
                    while chunk := g.read(chunk_size):
                        f.write(chunk)

        # Extract the concatenated tarball
        extract_archive(path)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        fig, axes = plt.subplots(ncols=self.seasons, squeeze=False, figsize=(4 * self.seasons, 4))
        num_bands = self.metadata[self.split]["num_bands"]
        rgb_bands = self.metadata[self.split]["rgb_bands"]

        for i in range(self.seasons):
            image = sample["image"][i * num_bands : (i + 1) * num_bands].byte()

            image = image[rgb_bands].permute(1, 2, 0)
            axes[0, i].imshow(image)
            axes[0, i].axis("off")

            if show_titles:
                axes[0, i].set_title(f"Split {self.split}, Season {i + 1}")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    def collate_fn(self, batch):
        """
        We need to define how tensors of different size (different number of objects per image) are going to be batched
        """
        if not self.configs["timeseries"]:
            labels = []
            images = []
            for data in batch:
                image, label = data
                label = torch.from_numpy(np.asarray([label]))
                label = einops.repeat(label, "c -> t c", t=self.seasons)
                images.append(image)
                labels.append(label)
            labels = torch.cat(labels, dim=0).squeeze()
            images = torch.cat(images, dim=0)
            if self.configs["keep_batch_size"]:
                indices = random.sample(range(0, images.shape[0] - 1), self.configs["batch_size"])
                images = images[indices, :, :, :]
                labels = labels[indices]
            return images, labels
        elif self.configs["stack"]:
            labels = []
            images = []
            for data in batch:
                image, label = data
                image = einops.rearrange(image, "t c h w -> (t c) h w")
                labels.append(label)
            labels = torch.from_numpy(np.asarray(labels))
            images = torch.cat(images)
            return images, labels
        else:
            return batch
