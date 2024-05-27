import builtins
import os
import random
import time

import einops
import numpy as np
import pyjson5 as json
import torch
from tqdm import tqdm

import utilities.model_utilities as model_utils
import utilities.utils as utils
import wandb
from utilities import distributed_utils as dist_utils
from utilities.model_utilities import adjust_learning_rate, get_current_learning_rate


def train_epoch_spectral(loader, mae, optimizer, epoch, configs, scaler):
    mae.train()
    configs["num_steps_per_epoch"] = configs["num_samples_per_epoch"] // configs["batch_size"]
    num_steps_per_epoch = configs["num_steps_per_epoch"]
    modality_dictionary = {v: k for k, v in configs["modality_channels"].items()}
    if configs["distributed"]:
        num_steps_per_epoch = num_steps_per_epoch // configs["world_size"]
        device = configs["local_rank"]
        disable = not dist_utils.is_global_master(configs)
    else:
        device = configs["device"]
        disable = False

    iterators = {}
    for dataloader, dataset_name in list(zip(loader, configs["train_datasets"])):
        iterators[dataset_name] = dataloader.__iter__()

    batches_per_dataset = {}
    loss_per_dataset = {}
    data_loading_time_per_dataset = {}

    # Set up gradient accumulation
    if configs["accumulate_gradients"] is not None:
        batches_to_accumulate = configs["accumulate_gradients"]

    running_loss = 0.0
    number_of_batches = 0.0
    for idx in tqdm(range(num_steps_per_epoch), disable=disable):
        if configs["dataset"] == "all":
            # Select dataset to operate on
            available_datasets = configs["train_datasets"]
            dataset_indices = list(range(len(available_datasets)))

            # Add option for weighted dataset selection
            if configs["dataset_probabilities"] is None:
                dataset_name, dataset_index = random.choice(list(zip(available_datasets, dataset_indices)))
            else:
                choice = random.choices(
                    list(zip(available_datasets, dataset_indices)), weights=configs["dataset_probabilities"], k=1
                )
                dataset_name, dataset_index = choice[0]

            train_loader = iterators[dataset_name]

            available_modalities = configs["dataset_modality_index"][dataset_name]

            num_masked_channels = random.randint(0, len(available_modalities) - 1)

            # mask_channels = int(num_masked_channels*len(available_modalities))
            channels_to_select = len(available_modalities) - num_masked_channels

            desired_channels = random.sample(list(available_modalities.keys()), channels_to_select)
            desired_indices = []
            spectral_keys = []
            for modality in desired_channels:
                desired_indices.append(available_modalities[modality])
                spectral_keys.append(int(modality_dictionary[modality]))

            if dataset_name not in data_loading_time_per_dataset:
                # [Running data loading time, Number of dataset iterations]
                data_loading_time_per_dataset[dataset_name] = [0.0, 0]
                loss_per_dataset[dataset_name] = 0.0

            if dataset_name in batches_per_dataset:
                batches_per_dataset[dataset_name] += 1
            else:
                batches_per_dataset[dataset_name] = 1
        start_time = time.time()
        batch = train_loader.__next__()

        end_time = time.time()
        data_loading_time_per_dataset[dataset_name][0] += end_time - start_time
        data_loading_time_per_dataset[dataset_name][1] += 1
            
        with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
            if (
                configs["accumulate_gradients"] is None
                or (idx + 1) % batches_to_accumulate == 0
                or (idx + 1) == num_steps_per_epoch
            ):
                # we use a per iteration (instead of per epoch) lr scheduler as done in official MAE implementation
                adjust_learning_rate(optimizer, idx / num_steps_per_epoch + epoch, configs)

            image, _ = batch
            image = image[:, desired_indices, :, :]

            image = image.to(device, non_blocking=True)

            if configs["dataset"] == "all":
                loss, predicted_pixels, reconstruction_data = mae((image, spectral_keys))
            else:
                loss, _ = mae(image)

        running_loss += loss.item()
        loss_per_dataset[dataset_name] += loss.item()
        number_of_batches += 1

        if idx % 100 == 0:
            if not configs["distributed"] or dist_utils.is_global_master(configs):
                log_dict = {"Epoch": epoch, "Iteration": idx, "train loss": running_loss / number_of_batches}
                running_loss = 0.0
                number_of_batches = 0.0
                for key in loss_per_dataset:
                    log_dict[key + " loss: "] = loss_per_dataset[key] / batches_per_dataset[key]
                for key in data_loading_time_per_dataset:
                    log_dict["Avg data loading time for: " + key] = (
                        data_loading_time_per_dataset[key][0] / data_loading_time_per_dataset[key][1]
                    )

                log_dict["Current Learning Rate"] = get_current_learning_rate(optimizer)

                if configs["wandb"]:
                    wandb.log(log_dict)
                else:
                    print(log_dict)

        # Scale loss according to gradient accumulation
        if configs["accumulate_gradients"] is not None:
            loss = loss / batches_to_accumulate

        if configs["mixed_precision"]:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # If gradient accumulation is enabled, update weights every batches_to_accumulate iterations.
        if (
            configs["accumulate_gradients"] is None
            or (idx + 1) % batches_to_accumulate == 0
            or (idx + 1) == num_steps_per_epoch
        ):
            if configs["mixed_precision"]:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
    print("=" * 20)
    print("Epoch sampling statistics")
    print(batches_per_dataset)
    print("=" * 20)


def train(configs):
    print("=" * 20)
    print("Initializing MAE")
    print("Training on the following datasets:")
    print(configs["train_datasets"])
    print("=" * 20)

    modality_channels = configs["modality_channels"]
    int_modality_channels = {}
    for key in modality_channels:
        int_modality_channels[int(key)] = modality_channels[key]
    configs["modality_channels"] = int_modality_channels

    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Setup distributed mode
    if configs["seed"] is not None:
        dist_utils.seed(configs)

    if configs["distributed"]:
        configs = dist_utils.init_distributed(configs)
        print("setting device for rank: ", configs["rank"])
        print("setting device for local rank: ", configs["local_rank"])
        print("Current device: ", torch.cuda.current_device())
        print("Total count: ", torch.cuda.device_count())
        print("world size ", configs["world_size"])

        torch.cuda.set_device(configs["local_rank"])

        if dist_utils.is_global_master(configs):
            id = wandb.util.generate_id()
            configs["wandb_id"] = id
            wandb.init(
                project=configs["wandb_project"], entity=configs["wandb_entity"], config=configs, id=id, resume="allow"
            )
    train_loader, _, _ = utils.create_dataloaders(configs)

    # Load all stats for denormalization when logging reconstructed images
    stats_config_path = "configs/stats/stats.json"
    stats_config = utils.load_config_file(stats_config_path)
    configs["all_stats"] = stats_config

    base_model = model_utils.create_model(configs)

    # Calculate effective batch size
    if configs["accumulate_gradients"] is None:
        accumulated_batches = 1
    else:
        accumulated_batches = configs["accumulate_gradients"]

    if not configs["distributed"]:
        world_size = 1
    else:
        world_size = configs["world_size"]

    # Scale learning rate linearly ( in regards to the effective batch size)
    configs["lr"] = configs["lr"] * accumulated_batches * world_size
    print("=" * 20)
    print("Scaled Learning Rate: ", configs["lr"])
    print("=" * 20)

    optimizer = utils.create_optimizer(configs)(
        base_model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )
    if not configs["distributed"]:
        base_model.to(configs["device"])
    else:
        base_model.to(configs["local_rank"])
        print("=" * 20)
        print("Initializing model for local_rank: ", configs["local_rank"])
        print("=" * 20)
        base_model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[configs["local_rank"]],
            output_device=configs["local_rank"],
            find_unused_parameters=True,
        )
    if not configs["distributed"] or dist_utils.is_global_master(configs):
        if configs["wandb"]:
            wandb.watch(base_model)

    mae = base_model  # torch.compile(base_model)
    if configs["start_epoch"] is None:
        start_epoch = 0
    else:
        start_epoch = configs["start_epoch"]
    for epoch in range(start_epoch, configs["epochs"]):
        if configs["spectral_mae"]:
            train_epoch_spectral(train_loader, mae, optimizer, epoch, configs, scaler)
        else:
            # Raise error if not implemented
            raise NotImplementedError
        if epoch % 1 == 0:
            if not configs["distributed"]:
                torch.save(base_model.state_dict(), os.path.join(configs["checkpoint_path"], "mae_" + str(epoch) + ".pt"))
                torch.save(base_model.encoder, os.path.join(configs["checkpoint_path"], "vit_" + str(epoch) + ".pt"))
            else:
                torch.save(
                    base_model.module.state_dict(), os.path.join(configs["checkpoint_path"], "mae_" + str(epoch) + ".pt")
                )
                torch.save(
                    base_model.module.encoder, os.path.join(configs["checkpoint_path"], "vit_" + str(epoch) + ".pt")
                )
    if not configs["distributed"]:
        torch.save(
            base_model.encoder.state_dict(),
            os.path.join(configs["checkpoint_path"], "mae_vit_" + str(configs["epochs"]) + ".pt"),
        )
        torch.save(
            base_model.encoder, os.path.join(configs["checkpoint_path"], "trained_vit_" + str(configs["epochs"]) + ".pt")
        )
    else:
        if dist_utils.is_global_master(configs):
            wandb.finish()
            torch.save(
                base_model.module.encoder.state_dict(),
                os.path.join(configs["checkpoint_path"], "mae_vit_" + str(configs["epochs"]) + ".pt"),
            )
            torch.save(
                base_model.module.encoder,
                os.path.join(configs["checkpoint_path"], "trained_vit_" + str(configs["epochs"]) + ".pt"),
            )
