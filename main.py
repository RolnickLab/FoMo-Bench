import json
import os
import pprint
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import utilities
import utilities.utils as utils
import argparse

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def seed_everything(seed):
    print("Setting seed to {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def store_experiment_id(configs):
    json.dump(
        {"run_id": configs["wandb_run_id"]},
        open(os.path.join(configs["checkpoint_path"], "id.json"), "w"),
    )


def init_wandb(configs):
    if configs["wandb_id_resume"] is None:
        id = wandb.util.generate_id()
    else:
        id = configs["wandb_id_resume"]
    wandb.init(
        project=configs["wandb_project"],
        entity=configs["wandb_entity"],
        config=configs,
        id=id,
        resume="allow",
    )
    run = wandb.run
    name = run.name
    configs["wandb_run_name"] = name
    configs["wandb_run_id"] = id
    if "checkpoint_path" not in configs.keys():
        checkpoint_path = utils.create_checkpoint_path(configs)
        configs["checkpoint_path"] = checkpoint_path
    store_experiment_id(configs)
    return configs


def init_offline_experiment(configs):
    configs["wandb_run_name"] = "offline"
    configs["wandb_run_id"] = "None"
    if "checkpoint_path" not in configs.keys():
        checkpoint_path = utils.create_checkpoint_path(configs)
        configs["checkpoint_path"] = checkpoint_path
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--method_config", default=None)
    parser.add_argument("--augmentation_config", default=None)
    parser.add_argument("--wandb_id_resume", default=None)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    # Setup configurations
    configs = utils.load_configs(args)
    pprint.pprint(configs)
    if args.seed is not None:
        configs["seed"] = args.seed
    seed_everything(configs["seed"])

    # Setup wandb
    if configs["wandb"] and not configs["distributed"]:
        configs = init_wandb(configs)
    else:
        configs = init_offline_experiment(configs)

    trainer, tester = utils.create_procedures(configs)

    if configs["phase"] == "train":
        trainer(configs)
    if tester is not None:
        _, _, loader = utils.create_dataloaders(configs)
        if isinstance(loader, list):
            dataset_names = configs['dataset_names'].split(',')
            for i, sub_loader in enumerate(loader):
                tester(configs, loader=sub_loader, phase="test", dataset_name=dataset_names[i])
        else:
            tester(configs, loader=loader, phase="test")
