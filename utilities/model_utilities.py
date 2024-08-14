import math
import sys

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from torchsummary import summary
import einops
from timm.models.layers import PatchEmbed
from model_zoo import multimodal_mae, segformer, upernet, pointnet, pointnet2, point_transformer
import yaml
from pathlib import Path

# from model_zoo import multimodal_mae, segformer, upernet, pointnet, pointnet2, point_transformer


class SupFoundation(nn.Module):
    def __init__(self, configs):
        super(SupFoundation, self).__init__()
        self.configs = configs
        configs["task"] = "classification"
        print("Creating model for classification task")
        self.base_model = create_classifier(configs)
        self.base_model.fc = nn.Identity()
        self.fc = {}
        for dataset in configs["train_datasets"]:
            self.fc[dataset] = nn.Linear(2048, configs["total_num_classes"][dataset])

    def forward(self, x, dataset):
        x = self.base_model(x)
        x = self.fc[dataset](x)
        return x


def create_presto(configs):
    raise NotImplementedError


def create_classifier(configs):
    if "polynet_evaluation" in configs and configs["polynet_evaluation"]:
        model = torch.load(configs["resnet_star_checkpoint"], map_location=torch.device("cpu"))
        model = model.base_model
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, configs["num_classes"])
        print("Loaded ResNet* model")
        return model
    if configs["backbone"].lower() == "presto":
        model = create_presto(configs)
    else:
        model = timm.create_model(
            configs["backbone"].lower(),
            pretrained=configs["pretrained"],
            in_chans=configs["in_channels"],
            num_classes=configs["num_classes"],
        )
        if "vit" in configs["backbone"] and configs["change_finetuning_resolution"] is not None:
            patch_size = model.patch_embed.patch_size
            img_size = configs["change_finetuning_resolution"]

            model.patch_embed = timm.layers.PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=configs["in_channels"],
                embed_dim=model.patch_embed.proj.out_channels,
                bias=True  # ,  # disable bias if pre-norm is used (e.g. CLIP)
                # dynamic_img_pad=False,
            )
            model.pos_embed = nn.Parameter(
                torch.randn(1, model.patch_embed.num_patches + 1, model.patch_embed.proj.out_channels) * 0.02
            )
            print("Adapted finetuning tokenizer")
            print(model)
    return model


def create_segmentor(configs):
    if configs["architecture"].lower() == "unet":
        model = smp.Unet(
            encoder_name=configs["backbone"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet"
            if configs["pretrained"]
            else None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=configs["in_channels"],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=configs["num_classes"],  # model output channels (number of classes in your dataset)
        )
    elif configs["architecture"].lower() == "deeplab":
        model = smp.DeepLabV3Plus(
            encoder_name=configs["backbone"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet"
            if configs["pretrained"]
            else None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=configs["in_channels"],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=configs["num_classes"],  # model output channels (number of classes in your dataset)
        )
    elif configs["architecture"].lower() == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=configs["backbone"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet"
            if configs["pretrained"]
            else None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=configs["in_channels"],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=configs["num_classes"],  # model output channels (number of classes in your dataset)
        )
    elif configs["architecture"].lower() == "upernet":
        model = upernet.UperNet(configs)
    elif configs["architecture"].lower() == "segformer":
        model = segformer.Segformer(configs)
    elif configs["architecture"].lower() == "presto":
        model = create_presto(configs)
    else:
        print("Model not supported.")
        exit(3)
    return model


def create_point_segmentor(configs):
    if configs["architecture"].lower() == "pointnet":
        if configs["pretrained_model_path"]:
            print("Loading checkpoint from: ", configs["pretrained_model_path"])
            backbone = torch.load(configs["pretrained_model_path"], map_location="cpu").backbone
            for param in backbone.parameters():
                if configs["finetune_backbone"]:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            model = FinetunerPCSeg(backbone, configs)
        else:
            model = pointnet.PointNet(nb_classes=configs["num_classes"])

    elif configs["architecture"].lower() == "pointnet2":
        model = pointnet2.PointNet2(nb_classes=configs["num_classes"])
    elif configs["architecture"].lower() == "point_transformer":
        model = point_transformer.PointTransformer(
            in_channels=3, out_channels=configs["num_classes"], dim_model=[32, 64, 128, 256, 512], k=16
        )
    else:
        print("Model not supported.")
        exit(3)
    return model


def create_detector(configs):
    if configs["architecture"].lower() == "fasterrcnn":
        if configs["backbone"].lower() == "resnet50":
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            if configs["num_classes"] == 1:
                # Faster RCNN needs actually 2 classes to discriminate
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=configs["num_classes"] + 1)
            else:
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=configs["num_classes"])
        elif configs["backbone"].lower() == "dinov2":
            from transformers import Dinov2Model, Dinov2Config
            from model_zoo.faster_rcnn.faster_rcnn import FasterRCNN
            from torchvision.models.detection.anchor_utils import AnchorGenerator
            from torchvision.ops import MultiScaleRoIAlign

            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),)
            backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base")
            backbone = Dinov2Model(backbone_config)
            backbone = FinetunerDetection(backbone, configs)
            backbone.out_channels = configs["out_channels"]

            anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
            roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

            if configs["num_classes"] == 1:
                nb_classes = configs["num_classes"] + 1
            else:
                nb_classes = configs["num_classes"]

            model = FasterRCNN(
                backbone=backbone,
                num_classes=nb_classes,
                min_size=configs["change_finetuning_resolution"],
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                batch_size=configs["batch_size"],
                out_channels=configs["out_channels"],
                output_size=configs["output_size"],
                box_score_thresh=configs["conf_thresh"],
                box_nms_thresh=configs["iou_thresh"],
            )
        elif configs["backbone"].lower() == "resnet50_star":
            from torchvision.models.detection.faster_rcnn import FasterRCNN
            from torchvision.models.detection.anchor_utils import AnchorGenerator
            from torchvision.ops import MultiScaleRoIAlign

            if configs["pretrained_model_path"]:
                print("Loading checkpoint from: ", configs["pretrained_model_path"])
                backbone = torch.load(configs["pretrained_model_path"], map_location="cpu").backbone
                for param in backbone.parameters():
                    if configs["finetune_backbone"]:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
                aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)  # Notation for BackboneWithFPN

                anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

                if configs["num_classes"] == 1:
                    nb_classes = configs["num_classes"] + 1
                else:
                    nb_classes = configs["num_classes"]

                model = FasterRCNN(
                    backbone=backbone,
                    num_classes=nb_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_score_thresh=configs["conf_thresh"],
                    box_nms_thresh=configs["iou_thresh"],
                )
            else:
                raise Exception("Backbone {} requires a pretained model path.".format(configs["backbone"]))

        else:
            print("Backbone {} not supported.".format(configs["backbone"]))
            exit(3)

    elif configs["architecture"].lower() == "retinanet":
        if configs["backbone"].lower() == "resnet50":
            from torchvision.models.detection import retinanet_resnet50_fpn_v2
            from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
            from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead

            model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT, min_size=224)

            # replace classification layer
            out_channels = model.head.classification_head.conv[0].out_channels
            num_anchors = model.head.classification_head.num_anchors

            if configs["num_classes"] == 1:
                num_classes = configs["num_classes"] + 1
            else:
                num_classes = configs["num_classes"]
            model.head.classification_head.num_classes = num_classes
            cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
            # assign cls head to model
            model.head.classification_head.cls_logits = cls_logits
        else:
            print("Backbone {} not supported.".format(configs["backbone"]))
            exit(3)

    elif configs["architecture"].lower() == "yolov5":
        model = torch.hub.load(
            "ultralytics/yolov5", "yolov5x", pretrained=False, autoshape=False, classes=configs["num_classes"]
        )
        hyp_path = Path.cwd() / "utilities" / "yolov5" / "hyp.scratch-low.yaml"
        with open(hyp_path, "r") as stream:
            hyp = yaml.safe_load(stream)
        model.hyp = hyp
    elif configs["architecture"].lower() == "yolos":
        from transformers import AutoModelForObjectDetection

        # Note: YOLOS intentionnaly add 1 class for "other" (already included in multi class detection)
        if configs["num_classes"] > 1:
            model = AutoModelForObjectDetection.from_pretrained(
                "hustvl/yolos-small", num_labels=configs["num_classes"] - 1, ignore_mismatched_sizes=True
            )
        else:
            model = AutoModelForObjectDetection.from_pretrained(
                "hustvl/yolos-small", num_labels=configs["num_classes"], ignore_mismatched_sizes=True
            )
    else:
        print("Model {} not supported.".format(configs["architecture"]))
        exit(3)
    return model


class FinetunerDetection(nn.Module):
    def __init__(self, encoder, configs=None, pool=False):
        super().__init__()
        self.configs = configs
        self.model = encoder
        self.pool = pool
        if configs["backbone"] == "dinov2":
            self.encoder_mlp_in_features = self.configs["encoder_in_features"]
        else:
            self.encoder_mlp_in_features = encoder.mlp_head.in_features * (configs["in_channels"])
        if not self.pool:
            if configs["deconv_decoder"]:
                self.head = nn.Sequential(
                    nn.Conv2d(self.encoder_mlp_in_features, 512, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, configs["out_channels"], kernel_size=1),
                )
            else:
                self.head = nn.Conv2d(self.encoder_mlp_in_features, configs["out_channels"], kernel_size=1)
        else:
            self.head = nn.Linear(
                self.encoder_mlp_in_features,
                configs["out_channels"] * configs["output_size"] * configs["output_size"],
            )
        if configs["backbone"] == "dinov2":
            self.model.encoder.layer[-1].mlp.fc2.in_features = nn.Identity()
        else:
            self.model.mlp_head = nn.Identity()

    def forward(self, x):
        img_size = self.configs["output_size"]

        if self.configs["backbone"] == "dinov2":
            x = self.model(x)["last_hidden_state"]
            if self.pool == False:
                x = x.view(
                    self.configs["batch_size"],
                    self.configs["finetuning_patch_size"],
                    self.configs["finetuning_patch_size"],
                    -1,
                ).permute(0, 3, 1, 2)
                x = nn.functional.interpolate(x, (img_size, img_size))
        else:
            x, keys = x
            x = self.model((x, keys), pool=self.pool)
            if self.pool == False:
                GS = img_size // self.configs["finetuning_patch_size"]
                x = einops.rearrange(x, "b (k h w) c -> b (c k) h w", h=GS, w=GS, k=len(keys))
                upsample = nn.Upsample(size=(img_size, img_size), mode="bilinear")
                x = upsample(x)
        return self.head(x)


class FinetunerSegmentation(nn.Module):
    def __init__(self, encoder, configs=None, pool=False):
        super().__init__()
        self.configs = configs
        self.model = encoder
        self.pool = pool
        if not self.pool:
            if configs["deconv_decoder"]:
                self.head = nn.Sequential(
                    nn.Conv2d(
                        encoder.mlp_head.in_features * (configs["in_channels"]), 512, kernel_size=3, stride=2, padding=1
                    ),
                    nn.ReLU(),
                    nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, configs["num_classes"], kernel_size=1),
                )
            else:
                self.head = nn.Conv2d(
                    encoder.mlp_head.in_features * (configs["in_channels"]), configs["num_classes"], kernel_size=1
                )
        else:
            self.head = nn.Linear(
                encoder.mlp_head.in_features * (configs["in_channels"]),
                configs["num_classes"] * configs["image_size"] * configs["image_size"],
            )
        self.model.mlp_head = nn.Identity()

    def forward(self, x):
        x, keys = x
        img_size = self.configs["image_size"]
        GS = img_size // self.configs["finetuning_patch_size"]
        x = self.model((x, keys), pool=self.pool)
        if self.pool == False:
            x = einops.rearrange(x, "b (k h w) c -> b (c k) h w", h=GS, w=GS, k=len(keys))
            upsample = nn.Upsample(size=(img_size, img_size), mode="bilinear")
            x = upsample(x)
        x = self.head(x)
        return x


class FinetunerClassification(nn.Module):
    def __init__(self, encoder, configs=None, pool=True):
        super().__init__()
        self.configs = configs
        self.model = encoder
        self.pool = pool
        self.head = nn.Linear(encoder.mlp_head.in_features, configs["num_classes"])
        self.model.mlp_head = nn.Identity()

    def forward(self, x):
        x, keys = x
        x = self.model((x, keys), pool=self.pool)
        x = self.head(x)
        return x


class FinetunerPCSeg(nn.Module):
    def __init__(self, encoder, configs=None):
        super().__init__()
        self.configs = configs
        self.model = encoder
        self.classifier = nn.Linear(encoder.classifier.in_features, configs["num_classes"])
        self.model.classifier = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


def create_model(configs):
    if "fully_finetune" not in configs:
        fully_finetune = False
    else:
        fully_finetune = configs["fully_finetune"]

    if not fully_finetune and not configs["linear_evaluation"]:
        if configs["task"] == "classification":
            model = create_classifier(configs=configs)
        elif configs["task"] == "segmentation":
            model = create_segmentor(configs=configs)
        elif configs["task"] == "point_segmentation":
            model = create_point_segmentor(configs=configs)
        elif configs["task"] == "detection":
            model = create_detector(configs=configs)
        elif configs["task"] == "supervised_foundation":
            model = SupFoundation(configs)
        elif configs["task"] == "mae":
            if configs["dataset"] == "all":
                if not configs["spectral_mae"]:
                    v = multimodal_mae.ViT(
                        image_size=configs["image_size"],
                        patch_size=configs["patch_size"],
                        channels=configs["modality_channels"],
                        num_classes=configs["num_classes"],
                        dim=configs["dim"],
                        depth=configs["depth"],
                        heads=configs["heads"],
                        mlp_dim=configs["mlp_dim"],
                    )
                    model = multimodal_mae.MultiModalMAE(
                        encoder=v,
                        masking_ratio=configs["masked_ratio"],  # the paper recommended 75% masked patches
                        decoder_dim=configs["decoder_dim"],  # paper showed good results with just 512
                        decoder_depth=configs["decoder_depth"],  # anywhere from 1 to 8
                        decoder_heads=configs["decoder_heads"],  # attention heads for decoder
                    )
                else:
                    v = multimodal_mae.MultiSpectralViT(
                        image_size=configs["image_size"],
                        patch_size=configs["patch_size"],
                        channels=1,
                        num_classes=configs["num_classes"],
                        dim=configs["dim"],
                        depth=configs["depth"],
                        heads=configs["heads"],
                        mlp_dim=configs["mlp_dim"],
                        configs=configs,
                    )
                    model = multimodal_mae.MultiSpectralMAE(
                        encoder=v,
                        masking_ratio=configs["masked_ratio"],  # the paper recommended 75% masked patches
                        decoder_dim=configs["decoder_dim"],  # paper showed good results with just 512
                        decoder_depth=configs["decoder_depth"],  # anywhere from 1 to 8
                        decoder_heads=configs["decoder_heads"],  # attention heads for decoder
                        configs=configs,
                    )
        else:
            print("Model for task: ", configs["task"], " is not implemented yet.")
            exit(3)

    if configs["resume_checkpoint_path"] is not None and not (configs["linear_evaluation"] or fully_finetune):
        print("Loading checkpoint from: ", configs["resume_checkpoint_path"])
        if configs["dataset"] != "supervised_foundation_cls":
            model.load_state_dict(torch.load(configs["resume_checkpoint_path"], map_location="cpu"))
        else:
            state_dict = torch.load(configs["resume_checkpoint_path"], map_location="cpu")
            model.base_model.load_state_dict(state_dict["base_model"])
            for key in state_dict["fc"]:
                model.fc[key].load_state_dict(state_dict["fc"][key])
            print("Loaded SupervisedFoundationModel checkpoint from: ", configs["resume_checkpoint_path"])

    if configs["linear_evaluation"] or fully_finetune:
        if configs["pretrained_model_path"] is not None:
            print("Loading checkpoint from: ", configs["pretrained_model_path"])
            model = torch.load(configs["pretrained_model_path"], map_location="cpu")

            if configs["linear_evaluation"]:
                print("Linear evaluation: Freezing all layers except the classification head.")
                for param in model.parameters():
                    param.requires_grad = False
            elif fully_finetune:
                print("Fully finetuning: Enable gradients for all layers.")
                for param in model.parameters():
                    param.requires_grad = True

            model.cls_token.requires_grad = True

            if configs["change_finetuning_resolution"] is not None:
                patch_size = configs["finetuning_patch_size"]
                num_patches = (configs["change_finetuning_resolution"] // patch_size) * (
                    configs["change_finetuning_resolution"] // patch_size
                )
                model.num_patches = num_patches
                assert (
                    configs["change_finetuning_resolution"] % patch_size == 0
                ), "Image dimensions must be divisible by the patch size."
                model.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, model.pos_embedding.shape[-1]))
                print("New positional embeddings: ")
                print(model.pos_embedding.shape)

            # Assume a transformer like model from MAE for now
            if configs["task"] == "classification":
                model = FinetunerClassification(model, configs)
            elif configs["task"] == "segmentation":
                print("=" * 20)
                print("Finetuning FoMo-Net for Segmentation")
                print("=" * 20)
                model = FinetunerSegmentation(model, configs)
            elif configs["task"] == "detection":
                if configs["architecture"].lower() == "fomonet":
                    # It uses a Faster RCNN head for finetuning
                    from model_zoo.faster_rcnn.faster_rcnn import FasterRCNN
                    from torchvision.models.detection.anchor_utils import AnchorGenerator
                    from torchvision.ops import MultiScaleRoIAlign

                    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
                    aspect_ratios = ((0.5, 1.0, 2.0),)
                    anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
                    roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
                    model = FinetunerDetection(model, configs)  # FoMo-Net backbone finetuning
                    model.out_channels = configs["out_channels"]

                    if configs["num_classes"] == 1:
                        nb_classes = configs["num_classes"] + 1
                    else:
                        nb_classes = configs["num_classes"]

                    model = FasterRCNN(
                        backbone=model,
                        num_classes=nb_classes,
                        min_size=configs["change_finetuning_resolution"],
                        rpn_anchor_generator=anchor_gen,
                        box_roi_pool=roi_pooler,
                        batch_size=configs["batch_size"],
                        out_channels=configs["out_channels"],
                        output_size=configs["output_size"],
                        finetuning_patch_size=configs["finetuning_patch_size"],
                        box_score_thresh=configs["conf_thresh"],
                        box_nms_thresh=configs["iou_thresh"],
                    )
                else:
                    import ipdb; ipdb.set_trace()
                    print("Finetuning on {} with model {} is not yet supported!".format(configs["task"], configs["architecture"]))
            else:
                print("Finetuning on ", configs["task"], " is not yet supported!")

            if configs["resume_checkpoint_path"] is not None:
                print("Resuming from FoMo checkpoint: ", configs["resume_checkpoint_path"])
                # model = torch.load(configs["resume_checkpoint_path"], map_location="cpu")
                model.load_state_dict(torch.load(configs["resume_checkpoint_path"], map_location="cpu"))

        else:
            print("Pretrained model path is None! Exiting!")
            exit(2)

    return model


def adjust_learning_rate(optimizer, epoch, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch <= configs["warmup_epochs"]:
        lr = configs["lr"] * epoch / configs["warmup_epochs"]
    else:
        lr = configs["min_lr"] + (configs["lr"] - configs["min_lr"]) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - configs["warmup_epochs"]) / (configs["epochs"] - configs["warmup_epochs"]))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_current_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
