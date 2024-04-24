import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed


def is_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ("LOCAL_RANK", "SLURM_LOCALID"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "SLURM_PROCID"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "SLURM_NTASKS"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def is_global_master(configs):
    return configs["rank"] == 0


def seed(configs):
    random.seed(configs["seed"])
    torch.manual_seed(configs["seed"])
    cudnn.deterministic = True
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )


def init_distributed(configs):
    if "SLURM_PROCID" in os.environ:
        configs["local_rank"], configs["rank"], configs["world_size"] = world_info_from_env()
        configs["num_workers"] = int(os.environ["SLURM_CPUS_PER_TASK"])
        os.environ["LOCAL_RANK"] = str(configs["local_rank"])
        os.environ["RANK"] = str(configs["rank"])
        os.environ["WORLD_SIZE"] = str(configs["world_size"])
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=configs["world_size"],
            rank=configs["rank"],
        )
    else:
        configs["local_rank"], _, _ = world_info_from_env()
        dist.init_process_group(backend="nccl")
        configs["world_size"] = dist.get_world_size()
        configs["rank"] = dist.get_rank()

    return configs
