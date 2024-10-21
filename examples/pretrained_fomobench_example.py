import torch.nn as nn
import torch
import timm
import argparse
import pyjson5 as json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default="configs/datasets/tallos.json")
    parser.add_argument("--model_config", default="configs/method/classification/convnext.json") #Backbone configuration file. See configs/method directory
    parser.add_argument("--checkpoint_path",default="YOUR_CHECKPOINT_PATH.pt")


    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path

    #load task specific configs
    with open(args.dataset_config, "r") as f:
        configs = json.load(f)

    #load model specific configs
    with open(args.model_config, "r") as f:
        model_configs = json.load(f)
    
    configs.update(model_configs)

    model = timm.create_model(
                configs["backbone"].lower(),
                pretrained=False,
                in_chans=configs["in_channels"],
                num_classes=configs["num_classes"],
            )

    #Load pretrained model

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
