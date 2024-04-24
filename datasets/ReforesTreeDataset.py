# Code is mostly based on https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/reforestree.py

import glob
import os
import pprint
import random
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path
import pickle

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pyjson5 as json
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import check_integrity, download_and_extract_archive, extract_archive
import albumentations as A
from albumentations.augmentations import Resize

from tqdm import tqdm
import einops

import utilities.augmentations
from utilities.utils import format_bboxes_voc_to_yolo


"""
Data loading for the ReforesTree dataset published in: 
 Reiersen, Gyri, et al. "ReforesTree: A Dataset for Estimating Tropical Forest Carbon Stock with Deep Learning and Aerial Imagery."
 Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 11. 2022.
"""


class ReforesTreeDataset(NonGeoDataset):
    """ReforesTree dataset.
    The `ReforesTree <https://github.com/gyrrei/ReforesTree>`__
    dataset contains drone imagery that can be used for tree crown detection,
    tree species classification and Aboveground Biomass (AGB) estimation.
    Dataset features:
    * 100 high resolution RGB drone images at 2 cm/pixel of size 4,000 x 4,000 px
    * more than 4,600 tree crown box annotations
    * tree crown matched with field measurements of diameter at breast height (DBH),
      and computed AGB and carbon values
    Dataset format:
    * images are three-channel pngs
    * annotations are csv file
    Dataset Classes:
    0. other
    1. banana
    2. cacao
    3. citrus
    4. fruit
    5. timber
    If you use this dataset in your research, please cite the following paper:
    * https://arxiv.org/abs/2201.11192
    .. versionadded:: 0.3
    """

    CLASSES = ["other", "banana", "cacao", "citrus", "fruit", "timber"]
    TEST_SITES = ["Leonor Aspiazu", "Manuel Macias"]
    TRAIN_SITES = ["Carlos Vera Arteaga", "Carlos Vera Guevara", "Flora Pluas", "Nestor Macias"]
    url = "https://zenodo.org/record/6813783/files/reforesTree.zip?download=1"

    md5 = "f6a4a1d8207aeaa5fbab7b21b683a302"
    zipfilename = "reforesTree.zip"

    def __init__(self, configs, mode="train"):
        """Initialize a new ReforesTree dataset instance.
        Args: (in configs)
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.cls = self.__class__
        self.configs = configs
        self.mode = mode
        self.root = Path(configs["root_path"])
        self.checksum = self.configs["checksum"]
        self.download = self.configs["download"]
        if self.configs["det_format"]:
            self.det_format = self.configs["det_format"]
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None

        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])
        self._verify()

        self.samples = []
        path_files = list((self.root / "tiles").glob("*/*.pkl"))
        for path_file in path_files:
            site = path_file.parent.name.split(" RGB")[0]
            if self.mode in ("train", "val") and site in self.cls.TRAIN_SITES:
                with open(path_file, "rb") as f:
                    self.samples.append(pickle.load(f))
            elif self.mode in ("test") and site in self.cls.TEST_SITES:
                with open(path_file, "rb") as f:
                    self.samples.append(pickle.load(f))
        self.samples = [item for sublist in self.samples for item in sublist]

        self.max_boxes = 0
        for sample in self.samples:
            with open(sample["labels"], "rb") as f:
                annots = pickle.load(f)
            bboxes = annots["boxes"]
            sample["boxes"] = bboxes
            sample["categories"] = annots["categories"]
            sample["AGB"] = annots["AGB"]
            sample["num_boxes"] = len(bboxes)
            if len(bboxes) > self.max_boxes:
                self.max_boxes = len(bboxes)

        # Filter images without boxes
        self.samples = [sample for sample in self.samples if len(sample["boxes"]) > 0]

        # For debugging
        # self.samples = self.samples[:100]

        if self.mode == "train":
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[: int(0.9 * len(self.samples))]

        elif self.mode == "val":
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[int(0.9 * len(self.samples)) :]

        self.num_examples = len(self.samples)
        print("Number of samples in split {}  = {}".format(self.mode, self.num_examples))

        # self.files = self._load_files(self.root)
        # self.annot_df = pd.read_csv(os.path.join(self.root, "mapping", "final_dataset.csv"))

        self.class2idx: Dict[str, int] = {c: i for i, c in enumerate(self.cls.CLASSES)}
        for i in range(len(self.samples)):
            self.samples[i]["cat_id"] = np.array([self.class2idx[cat] for cat in self.samples[i]["categories"]])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        sample = self.samples[index]

        with Image.open(sample["image"]) as fp:
            image = np.array(fp)
        bboxes = np.array(sample["boxes"])
        class_labels = sample["categories"]

        if self.det_format == "coco":
            # [x_min, y_min, x_max, y_max] -> [x_min, y_min, w, h]
            bboxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in bboxes]
        elif self.det_format == "yolo":
            img_size = image.shape[:2]
            # [x_min, y_min, x_max, y_max] -> [x_min, x_max, y_min, y_max] -> [x_center, y_center, w, h] in relative coords
            bboxes = [format_bboxes_voc_to_yolo([box[0], box[2], box[1], box[3]], img_size) for box in bboxes]
        if not self.configs["webdataset"]:
            if self.configs["augment"] and self.mode == "train":
                # image = einops.rearrange(image, 'c h w -> h w c')
                transform = self.augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
                image = transform["image"]
                bboxes = transform["bboxes"]
                class_labels = transform["class_labels"]
                image = einops.rearrange(image, "h w c -> c h w")

            if self.mode in ("val", "test") and "Resize" in list(self.configs["augmentations"].keys()):
                # image = einops.rearrange(image, 'c h w -> h w c')
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
        else:
            image = torch.from_numpy(image).float()

        # boxes, labels, agb = self._load_target(filepath)
        # temporarly
        bboxes = np.array(bboxes)
        if self.det_format == "coco":
            areas = bboxes[:, 2] * bboxes[:, 3]
        elif self.det_format == "yolo":
            # not required
            pass
        else:
            # pascal voc format
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
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

    def __len__(self) -> int:
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.samples)

    def _load_files(self, root: str) -> List[str]:
        """Return the paths of the files in the dataset.
        Args:
            root: root dir of dataset
        Returns:
            list of dicts containing paths for each pair of image, annotation
        """
        image_paths = sorted(glob.glob(os.path.join(root, "tiles", "**", "*.png")))

        return image_paths

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.
        Args:
            path: path to the image
        Returns:
            the image
        """
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.uint8]" = np.array(img)
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, filepath: str) -> Tuple[Tensor, ...]:
        """Load boxes and labels for a single image.
        Args:
            filepath: image tile filepath
        Returns:
            dictionary containing boxes, label, and agb value
        """
        tile_df = self.annot_df[self.annot_df["img_path"] == os.path.basename(filepath)]

        boxes = torch.Tensor(tile_df[["xmin", "ymin", "xmax", "ymax"]].values.tolist())
        labels = torch.Tensor([self.class2idx[label] for label in tile_df["group"].tolist()])
        agb = torch.Tensor(tile_df["AGB"].tolist())

        return boxes, labels, agb

    def _verify(self) -> None:
        """Checks the integrity of the dataset structure.
        Raises:
            RuntimeError: if dataset is not found in root or is corrupted
        """
        filepaths = [os.path.join(self.root, dir) for dir in ["tiles", "mapping"]]
        if all([os.path.exists(filepath) for filepath in filepaths]):
            return

        filepath = os.path.join(self.root, self.zipfilename)
        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            print(filepath)
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # else download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset and extract it.
        Raises:
            AssertionError: if the checksum does not match
        """
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.zipfilename,
            md5=self.md5 if self.checksum else None,
        )

    def plot(
        self,
        index: 0,
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
        sample = self.__getitem__(index)
        image = sample["image"].permute((1, 2, 0)).numpy()
        ncols = 1
        showing_predictions = "prediction_boxes" in sample
        if showing_predictions:
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if not showing_predictions:
            axs = [axs]

        axs[0].imshow(image)
        axs[0].axis("off")

        bboxes = [
            patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            for bbox in sample["boxes"].numpy()
        ]
        for bbox in bboxes:
            axs[0].add_patch(bbox)

        if show_titles:
            axs[0].set_title("Ground Truth")

        if showing_predictions:
            axs[1].imshow(image)
            axs[1].axis("off")

            pred_bboxes = [
                patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                for bbox in sample["prediction_boxes"].numpy()
            ]
            for bbox in pred_bboxes:
                axs[1].add_patch(bbox)

            if show_titles:
                axs[1].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        plt.savefig("ReforesTree_sample_" + str(index) + ".png")
        return fig
