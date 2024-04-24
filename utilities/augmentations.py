import albumentations as A
import torch_geometric.transforms as T


def get_augmentations(config):
    augmentations = config["augmentations"]
    task = config["task"]
    independend_aug = []
    for k, v in augmentations.items():
        if k == "RandomResizedCrop":
            aug = A.augmentations.RandomResizedCrop(
                height=v["value"], width=v["value"], p=v["p"], scale=tuple(v["scale"]), interpolation=v["interpolation"]
            )
        elif k == "Resize":
            aug = A.augmentations.Resize(height=v["value"], width=v["value"], p=v["p"])
        elif k == "ColorJitter":
            aug = A.augmentations.ColorJitter(
                brightness=v["value"][0],
                contrast=v["value"][1],
                saturation=v["value"][2],
                hue=v["value"][3],
                p=v["p"],
            )
        elif k == "HorizontalFlip":
            aug = A.augmentations.HorizontalFlip(p=v["p"])
        elif k == "VerticalFlip":
            aug = A.augmentations.VerticalFlip(p=v["p"])
        elif k == "RandomRotation":
            aug = A.augmentations.Rotate(p=v["p"])
        elif k == "GaussianBlur":
            aug = A.augmentations.GaussianBlur(sigma_limit=v["value"], p=v["p"])
        elif k == "ElasticTransform":
            aug = A.augmentations.ElasticTransform(p=v["p"])
        elif k == "Cutout":
            aug = A.augmentations.CoarseDropout(p=v["p"])
        elif k == "GaussianNoise":
            aug = A.augmentations.GaussNoise(p=v["p"])
        elif k == "MultNoise":
            aug = A.augmentations.MultiplicativeNoise(p=v["p"])
        elif k == "SamplePoints":
            aug = T.SamplePoints(num=v["num"], remove_faces=v["remove_faces"], include_normals=v["include_normals"])
        elif k == "RandomJitter":
            aug = T.RandomJitter(translate=v["translate"])
        elif k == "RandomRotate_x":
            aug = T.RandomRotate(degrees=v["degrees"], axis=0)
        elif k == "RandomRotate_y":
            aug = T.RandomRotate(degrees=v["degrees"], axis=1)
        elif k == "RandomRotate_z":
            aug = T.RandomRotate(degrees=v["degrees"], axis=2)
        else:
            print("Augmentation: ", k, " not supported!")
            exit(2)
        independend_aug.append(aug)
    if task == "detection":
        return A.Compose(
            independend_aug,
            bbox_params=A.BboxParams(format=config["det_format"], min_visibility=0.01, label_fields=["class_labels"]),
        )
    elif task == "point_segmentation":
        return T.Compose(independend_aug)
    return A.Compose(independend_aug)
