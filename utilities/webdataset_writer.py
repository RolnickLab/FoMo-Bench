import os
from pathlib import Path

import albumentations as A
import einops
import torch
import tqdm
import webdataset as wds

import utilities.utils as utils
import ray


def wds_write(configs):
    for mode in ["train", "val", "test"]:
        dataset = utils.load_dataset(configs, mode=mode)
        print("=" * 40)
        print("Creating shards for dataset: ", configs["dataset"])
        print("Mode: ", mode, " Size: ", len(dataset))
        print("=" * 40)

        if configs["max_sample_resolution"] is None:
            shard_path = Path(os.path.join(configs["root_path"], "webdataset", configs["dataset"], mode))
            shard_path.mkdir(parents=True, exist_ok=True)

            pattern = os.path.join(configs["root_path"], "webdataset", configs["dataset"], mode, f"sample-{mode}-%06d.tar")
        else:
            shard_path = Path(
                os.path.join(
                    configs["root_path"],
                    "webdataset" + "_" + str(configs["max_sample_resolution"]),
                    "webdataset",
                    configs["dataset"],
                    mode,
                )
            )
            shard_path.mkdir(parents=True, exist_ok=True)
            pattern = os.path.join(
                configs["root_path"],
                "webdataset" + "_" + str(configs["max_sample_resolution"]),
                "webdataset",
                configs["dataset"],
                mode,
                f"sample-{mode}-%06d.tar",
            )
        with wds.ShardWriter(pattern, maxcount=configs["max_samples_per_shard"]) as sink:
            for index, batch in enumerate(tqdm.tqdm(dataset)):
                if isinstance(batch, dict):
                    image = batch["image"]
                else:
                    (image, labels) = batch
                if configs["max_sample_resolution"] is not None:
                    image = image.permute(1, 2, 0).numpy()
                    resize = A.Compose(
                        [
                            A.augmentations.Resize(
                                height=configs["max_sample_resolution"], width=configs["max_sample_resolution"], p=1.0
                            )
                        ]
                    )
                    transform = resize(image=image)
                    image = transform["image"]
                    image = torch.from_numpy(einops.rearrange(image, "h w c -> c h w"))

                if isinstance(batch, dict):
                    labels_dict = {}
                    for key in batch:
                        if key != "image":
                            labels_dict[key] = batch[key]
                    sink.write({"__key__": "sample%06d" % index, "image.pth": image, "labels.pth": labels_dict})
                else:
                    sink.write({"__key__": "sample%06d" % index, "image.pth": image, "labels.pth": labels})


@ray.remote
def wds_write_ith_shard(configs, dataset, mode, i, n):
    if configs["max_sample_resolution"] is None:
        shard_path = Path(os.path.join(configs["root_path"], "webdataset", configs["dataset"], mode))
        shard_path.mkdir(parents=True, exist_ok=True)

        pattern = os.path.join(configs["root_path"], "webdataset", configs["dataset"], mode, f"sample-{mode}-{i}-%06d.tar")
    else:
        shard_path = Path(
            os.path.join(
                configs["root_path"],
                "webdataset" + "_" + str(configs["max_sample_resolution"]),
                "webdataset",
                configs["dataset"],
                mode,
            )
        )
        shard_path.mkdir(parents=True, exist_ok=True)
        pattern = os.path.join(
            configs["root_path"],
            "webdataset" + "_" + str(configs["max_sample_resolution"]),
            "webdataset",
            configs["dataset"],
            mode,
            f"sample-{mode}-{i}-%06d.tar",
        )

    with wds.ShardWriter(pattern, maxcount=configs["max_samples_per_shard"]) as sink:
        for index in tqdm.tqdm(range(i, len(dataset), n)):
            batch = dataset[index]
            if isinstance(batch, dict):
                image = batch["image"]
            else:
                (image, labels) = batch
            if configs["max_sample_resolution"] is not None:
                image = image.permute(1, 2, 0).numpy()
                resize = A.Compose(
                    [
                        A.augmentations.Resize(
                            height=configs["max_sample_resolution"], width=configs["max_sample_resolution"], p=1.0
                        )
                    ]
                )
                transform = resize(image=image)
                image = transform["image"]
                image = torch.from_numpy(einops.rearrange(image, "h w c -> c h w"))

            if isinstance(batch, dict):
                labels_dict = {}
                for key in batch:
                    if key != "image":
                        labels_dict[key] = batch[key]
                sink.write({"__key__": "sample%06d" % index, "image.pth": image, "labels.pth": labels_dict})
            else:
                sink.write({"__key__": "sample%06d" % index, "image.pth": image, "labels.pth": labels})


def wds_write_parallel(configs):
    ray.init()
    n = configs["webdataset_write_processes"]
    for mode in ["train", "val", "test"]:
        dataset = utils.load_dataset(configs, mode=mode)
        print("=" * 40)
        print("Creating shards for dataset: ", configs["dataset"])
        print("Mode: ", mode, " Size: ", len(dataset))
        print("=" * 40)

        ray.get([wds_write_ith_shard.remote(configs, dataset, mode, i, n) for i in range(n)])
