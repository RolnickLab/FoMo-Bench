from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation, AutoConfig
import torch
import torch.nn as nn

backbones = {
    "swin_base": "openmmlab/upernet-swin-base",
    "swin_tiny": "openmmlab/upernet-swin-tiny",
    "swin_small": "openmmlab/upernet-swin-small",
    "convnext_tiny": "openmmlab/upernet-convnext-tiny",
    "convnext_small": "openmmlab/upernet-convnext-small",
    "convnext_base": "openmmlab/upernet-convnext-base",
}


class UperNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config["backbone"] not in backbones:
            print("Backbone: ", config["backbone"], " Not supported!")
            exit(2)
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            backbones[config["backbone"]], num_labels=config["num_classes"], ignore_mismatched_sizes=True
        )
        if config["in_channels"] != 3:
            if "convnext" in config["backbone"]:
                out_channels = self.model.backbone.embeddings.patch_embeddings.out_channels
                kernel_size = self.model.backbone.embeddings.patch_embeddings.kernel_size
                stride = self.model.backbone.embeddings.patch_embeddings.stride
                self.model.backbone.embeddings.num_channels = config["in_channels"]
                self.model.config.backbone_config.num_channels = config["in_channels"]
                self.model.backbone.embeddings.patch_embeddings = nn.Conv2d(
                    config["in_channels"], out_channels, kernel_size=kernel_size, stride=stride
                )
            elif "swin" in config["backbone"]:
                out_channels = self.model.backbone.embeddings.patch_embeddings.projection.out_channels
                kernel_size = self.model.backbone.embeddings.patch_embeddings.projection.kernel_size
                stride = self.model.backbone.embeddings.patch_embeddings.projection.stride
                self.model.backbone.embeddings.patch_embeddings.num_channels = config["in_channels"]
                self.model.backbone.embeddings.patch_embeddings.projection = nn.Conv2d(
                    config["in_channels"], out_channels, kernel_size=kernel_size, stride=stride
                )

    def forward(self, x):
        x = self.model(x, return_dict=True)
        return x["logits"]


"""
#Example usage:
#================
config= {'in_channels':5,'num_classes':2,'backbone':'swin_tiny'}
k = torch.randn((4,config['in_channels'],120,120))
model = UperNet(config)
logits = model(k)
print(logits.shape)
"""
