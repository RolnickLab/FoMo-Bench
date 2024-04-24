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
import tqdm
import random
import utilities
import pickle
import rioxarray as rio


class RapidAI4EODataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode="train"):
        print("=" * 40)
        print("Initializing RapidAI4EO mode - ", mode)
        print("=" * 40)

        self.configs = configs
        self.root_path = os.path.join(configs["root_path"], "rapidai4eo")
        self.data_path = os.path.join(self.root_path, "imagery")
        self.labels_path = os.path.join(self.root_path, "labels")
        self.mode = mode
        self.modalities = []
        if configs["source"] == "planet":
            self.modalities = ["planet"]
        elif configs["source"] == "sentinel":
            self.modalities = ["sentinel"]
        elif configs["source"] == "all":
            self.modalities = ["sentinel", "planet"]
        else:
            print("Source: ", configs["source"], " not supported!")
            exit(3)
        if self.configs["augment"]:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None
        if self.configs["normalization"] == "standard":
            self.normalization = transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"])

        areas = os.listdir(self.data_path)
        count_sentinel = 0
        count_planet = 0
        both_sources = 0
        self.samples = []
        single_planet_images = 0.0

        test_areas = ["18E", "27E"]
        if configs["timeseries"]:
            time_str = "timeseries"
        else:
            time_str = "single_image"
        if not os.path.isfile(
            os.path.join(
                self.root_path,
                "samples_dict_frequency_" + str(self.configs["planet_frequency"]) + time_str + "_" + mode + ".pkl",
            )
        ):
            if mode == "val" and os.path.isfile(
                os.path.join(
                    self.root_path,
                    "samples_dict_frequency_" + str(self.configs["planet_frequency"]) + time_str + "_" + "train" + ".pkl",
                )
            ):
                with open(
                    os.path.join(
                        self.root_path,
                        "samples_dict_frequency_"
                        + str(self.configs["planet_frequency"])
                        + time_str
                        + "_"
                        + "train"
                        + ".pkl",
                    ),
                    "rb",
                ) as f:
                    self.samples = pickle.load(f)
            else:
                for area in areas:
                    subareas_path = os.path.join(self.data_path, area)
                    subareas = os.listdir(subareas_path)
                    print("Loading area:")
                    print(area)
                    for subarea in tqdm.tqdm(subareas):
                        if (subarea.split("-")[0] in test_areas and self.mode != "test") or (
                            subarea.split("-")[0] not in test_areas and self.mode == "test"
                        ):
                            continue
                        patches_path = os.path.join(subareas_path, subarea)
                        patches = os.listdir(patches_path)

                        for patch in patches:
                            patch_path = os.path.join(patches_path, patch)
                            planet_path = os.path.join(patch_path, "PF-SR")
                            sentinel_path = os.path.join(patch_path, "S2-SR")
                            planet_data = []
                            sentinel_data = []

                            if os.path.isdir(planet_path) and "planet" in self.modalities:
                                count_planet += 1
                                planet_series = os.listdir(planet_path)
                                for planet_sample in planet_series:
                                    single_planet_images += 1
                                    planet_sample_path = os.path.join(planet_path, planet_sample)
                                    planet_data.append(planet_sample_path)
                            if os.path.isdir(sentinel_path) and "sentinel" in self.modalities:
                                count_sentinel += 1
                                sentinel_series = os.listdir(sentinel_path)
                                for sentinel_sample in sentinel_series:
                                    sentinel_sample_path = os.path.join(sentinel_path, sentinel_sample)
                                    sentinel_data.append(sentinel_sample_path)

                            label_path = os.path.join(self.labels_path, area, subarea, "labels_" + patch + ".geojson")
                            if len(planet_data) == 0 and len(sentinel_data) == 0:
                                print("Patch has no data!")
                                print(patch_path)
                            elif len(planet_data) > 0 and len(sentinel_data) > 0:
                                both_sources += 1

                            valid_sample = False
                            if self.configs["source"] == "planet":
                                if len(planet_data) > 0:
                                    valid_sample = True
                            elif self.configs["source"] == "sentinel":
                                if len(sentinel_data) > 0:
                                    valid_sample = True
                            elif self.configs["source"] == "all":
                                if len(planet_data) > 0 and len(sentinel_data) > 0:
                                    valid_sample = True
                            else:
                                print("Requested source is not available!")
                                exit(2)
                            if not self.configs["timeseries"]:
                                for pl_index in range(0, len(planet_data), self.configs["planet_frequency"]):
                                    planet_sample = planet_data[pl_index]
                                    # for planet_sample in planet_data:
                                    if "sentinel" in self.modalities:
                                        year = planet_sample.split("/")[-1][:4]
                                        month = planet_sample.split("/")[-1][4:7]
                                        sentinel_sample = os.path.join(sentinel_path, year + month + ".tif")

                                        if not os.path.isfile(sentinel_sample):
                                            continue
                                    else:
                                        sentinel_sample = None
                                    if "Download" in planet_sample:
                                        continue
                                    sample = {"planet": planet_sample, "sentinel": sentinel_sample, "label": label_path}
                                    if not os.path.isfile(label_path):
                                        continue
                                    if valid_sample:
                                        self.samples.append(sample)
                            else:
                                sample = {
                                    "planet": np.sort(planet_data),
                                    "sentinel": np.sort(sentinel_data),
                                    "label": label_path,
                                }
                                if valid_sample:
                                    self.samples.append(sample)

                random.Random(999).shuffle(self.samples)

                with open(
                    os.path.join(
                        self.root_path,
                        "samples_dict_frequency_" + str(self.configs["planet_frequency"]) + time_str + "_" + mode + ".pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(self.samples, f)
        else:
            with open(
                os.path.join(
                    self.root_path,
                    "samples_dict_frequency_" + str(self.configs["planet_frequency"]) + time_str + "_" + mode + ".pkl",
                ),
                "rb",
            ) as f:
                self.samples = pickle.load(f)
        if self.mode == "train":
            self.samples = self.samples[: int(0.9 * len(self.samples))]
        elif self.mode == "val":
            self.samples = self.samples[int(0.9 * len(self.samples)) :]

        self.num_examples = len(self.samples)
        print("Mode: ", mode, "Number of examples: ", self.num_examples)
        print("Number of sentinel series: ", count_sentinel)
        print("Number of planet series: ", count_planet)
        print("Number of planet samples: ", single_planet_images)
        print("Number of samples with both sources: ", both_sources)

    def __len__(self):
        return self.num_examples

    def read_labels(self, label_path):
        label = json.load(open(label_path, "r"))["properties"]
        aggregated_labels = []
        # Add artificial surfaces
        artificial_surfaces = (
            label["clc_111"]
            + label["clc_112"]
            + label["clc_121"]
            + label["clc_122"]
            + label["clc_123"]
            + label["clc_124"]
            + label["clc_131"]
            + label["clc_132"]
            + label["clc_133"]
            + label["clc_141"]
            + label["clc_142"]
        )
        aggregated_labels.append(artificial_surfaces)

        # Add crops
        crops = [
            "clc_211",
            "clc_212",
            "clc_213",
            "clc_221",
            "clc_222",
            "clc_223",
            "clc_231",
            "clc_241",
            "clc_242",
            "clc_243",
            "clc_244",
        ]
        for crop_class in crops:
            aggregated_labels.append(label[crop_class])

        # Add forest related classes
        forests = [
            "clc_311",
            "clc_312",
            "clc_313",
            "clc_321",
            "clc_322",
            "clc_323",
            "clc_324",
            "clc_331",
            "clc_332",
            "clc_333",
            "clc_334",
            "clc_335",
        ]
        for forest_class in forests:
            aggregated_labels.append(label[forest_class])

        # Add wetlands
        wetlands = label["clc_411"] + label["clc_412"] + label["clc_421"] + label["clc_422"] + label["clc_423"]
        aggregated_labels.append(wetlands)

        # Add water bodies
        water_bodies = label["clc_511"] + label["clc_512"] + label["clc_521"] + label["clc_522"] + label["clc_523"]
        aggregated_labels.append(water_bodies)

        return np.asarray(aggregated_labels)

    def read_tif(self, tif_path, sentinel=False):
        tif = rio.open_rasterio(tif_path, engine="rasterio")

        tif = tif.rio.reproject(
            "EPSG:4326", shape=(self.configs["size"], self.configs["size"]), resampling=rasterio.enums.Resampling.bilinear
        )

        lat = tif.y.to_numpy()  # np.degrees(np.arcsin(tif.y / radius_earth)
        lon = tif.x.to_numpy()

        meta = {"lat": lat, "lon": lon}

        return tif.to_numpy(), meta

    def average_captions(self, tif_dict, temporal_resolution=""):
        if temporal_resolution == "planet":
            temporal_resolution = "planet_temporal_resolution"
        elif temporal_resolution == "sentinel":
            temporal_resolution = "sentinel_temporal_resolution"
        else:
            print("Uknown temporal resolution keyword! Exiting!")
            exit(2)
        number_of_captions_per_year = self.configs[temporal_resolution]
        flattened_dict = [item for sublist in list(tif_dict.values()) for item in sublist]
        number_of_captions_to_average = len(flattened_dict) // number_of_captions_per_year

        if number_of_captions_per_year == len(flattened_dict):
            # print(temporal_resolution,' Full series is returned')
            captions = []
            for file in flattened_dict:
                captions.append(self.read_tif(file))
            return captions
        else:
            mean_captions = []
            buffer = []
            for idx in range(len(flattened_dict)):
                buffer.append(self.read_tif(flattened_dict[idx]))
                if len(buffer) == number_of_captions_to_average:
                    means = np.mean(buffer.copy(), axis=0)
                    mean_captions.append(means)
                    buffer = []
                if len(mean_captions) == number_of_captions_per_year:
                    break

            return mean_captions

    def __getitem__(self, index):
        sample = self.samples[index]
        planet_paths = sample["planet"]
        sentinel_paths = sample["sentinel"]
        label_path = sample["label"]

        if self.configs["timeseries"]:
            # Calc means for planet data
            planet_dict = {}
            for planet_caption in planet_paths:
                caption_date = planet_caption.split("/")[-1][:-4]
                year_month_day = caption_date.split("-")
                year = year_month_day[0]
                month = year_month_day[1]
                if year in planet_dict:
                    if month in planet_dict[year]:
                        planet_dict[year][month].append(planet_caption)
                    else:
                        planet_dict[year][month] = [planet_caption]
                else:
                    planet_dict[year] = {}
                    planet_dict[year][month] = [planet_caption]
            planet_series = []
            for year in planet_dict:
                planet_series.extend(self.average_captions(planet_dict[year], temporal_resolution="planet"))
            planet_series = np.asarray(planet_series)

            # Read sentinel data
            if "sentinel" in self.modalities:
                sentinel_series = []
                sentinel_series.extend(self.average_captions({"2018": sentinel_paths}, temporal_resolution="sentinel"))

                sentinel_series = np.asarray(sentinel_series)

            labels = self.read_labels(label_path)
            if self.configs["source"] == "all":
                # Stack sources
                resize = A.Compose(
                    [A.augmentations.Resize(height=planet_series.shape[2], width=planet_series.shape[3], p=1.0)]
                )
                sentinel_series = einops.rearrange(sentinel_series, "t c h w -> h w (t c)")
                planet_series = einops.rearrange(planet_series, "t c h w -> (t c) h w")
                sentinel_transform = resize(image=sentinel_series)
                sentinel_series = sentinel_transform["image"]
                sentinel_series = einops.rearrange(sentinel_series, "h w c -> c h w")

                image = np.vstack((planet_series, sentinel_series))
            elif self.configs["source"] == "planet":
                planet_series = einops.rearrange(planet_series, "t c h w -> (t c) h w")
                image = planet_series
            elif self.configs["source"] == "sentinel":
                sentinel_series = einops.rearrange(sentinel_series, "t c h w ->(t c) h w ")
                image = sentinel_series

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

            return image, labels
        else:
            # In single image setting, planet and sentinel images are stacked
            planet_tif, meta = self.read_tif(planet_paths)

            planet_image = torch.from_numpy(planet_tif)
            if "sentinel" in self.modalities:
                sentinel_tif, meta_s2 = self.read_tif(sentinel_paths, sentinel=True)
                sentinel_image = torch.from_numpy(sentinel_tif.astype(float))
                image = torch.cat((planet_image, sentinel_image), dim=0).float()
                labels = {
                    "label": self.read_labels(label_path),
                    "coordinates": {
                        "lat": meta["lat"],
                        "lon": meta["lon"],
                        "lat_s2": meta_s2["lat"],
                        "lon_s2": meta_s2["lon"],
                    },
                }

            else:
                image = planet_image

                labels = {"label": self.read_labels(label_path), "coordinates": {"lat": meta["lat"], "lon": meta["lon"]}}
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

            return image, labels
