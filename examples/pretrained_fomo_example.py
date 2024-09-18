from model_zoo import multimodal_mae
import torch
import torch.nn as nn
import pyjson5 as json
import argparse


def construct_fomo_configs(args):
    '''
        Construct configurations for FoMo_1 model
    '''

    configs = {
        "image_size":args.image_size,
        "patch_size":args.patch_size,
        "dim":args.dim,
        "depth":args.depth,
        "heads":args.heads,
        "mlp_dim":args.mlp_dim,
        "num_classes":args.num_classes,
        "single_embedding_layer":True,
    }

    #Update configs with modality specific configurations as defined during pretraining

    modality_configs = json.load(open(args.modality_configs,'r'))
    configs.update(modality_configs)

    return configs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint_path",default="fomo_single_embedding_layer_weights.pt")
    parser.add_argument("--modality_configs", default="configs/datasets/fomo_pretraining_datasets.json")
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--dim", default=768, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--heads", default=12, type=int)
    parser.add_argument("--mlp_dim", default=2048, type=int)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--single_embedding_layer", default=True, type=bool)

    args = parser.parse_args()

    configs = construct_fomo_configs(args)   

    #Initialize FoMo model
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
    v.load_state_dict(torch.load( args.checkpoint_path,map_location='cpu'))
