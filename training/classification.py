import pprint

import numpy as np
import pyjson5 as json
import torch
import tqdm
import wandb
import os
import utilities.model_utilities as model_utils
import utilities.utils as utils
from utilities.model_utilities import adjust_learning_rate, get_current_learning_rate


def early_stopping(val_loss, best_loss, counter, patience=5):
    if val_loss < best_loss:
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        return True, counter
    else:
        return False, counter


def train_epoch(train_loader, model, optimizer, criterion, epoch, configs, scaler):
    if not configs["linear_evaluation"] and not configs["fully_finetune"]:
        model.train()
    else:
        model.train()

        modality_dictionary = {v: k for k, v in configs["modality_channels"].items()}
        available_modalities = configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        desired_indices = []
        for modality in available_modalities:
            spectral_keys.append(int(modality_dictionary[modality]))
            desired_indices.append(available_modalities[modality])

    if "polynet_evaluation" in configs and configs["polynet_evaluation"]:
        resnet_star_configs = json.load(open("configs/datasets/supervised_foundation_cls.json", "r"))

        polynet_evaluation = True
        modality_dictionary = {v: k for k, v in resnet_star_configs["modality_channels"].items()}
        available_modalities = resnet_star_configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        desired_indices = []
        for modality in available_modalities:
            spectral_keys.append(int(modality_dictionary[modality]))
            desired_indices.append(available_modalities[modality])
        print("Desired indices: ", desired_indices)
    else:
        polynet_evaluation = False

    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
        if "samples_per_epoch" in configs:
            if (idx + 1) * configs["batch_size"] > configs["samples_per_epoch"]:
                break
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
            image, label = batch

            image = image.to(configs["device"])
            label = label.to(configs["device"])

            if not configs["linear_evaluation"] and not configs["fully_finetune"]:
                if not polynet_evaluation:
                    out = model(image)
                else:
                    image = image[:, desired_indices, :, :]
                    out = model(image)
            else:
                image = image[:, desired_indices, :, :]

                out = model((image, spectral_keys))
            loss = criterion(out, label)
        if idx % 100 == 0:
            log_dict = {"Epoch": epoch, "Iteration": idx, "train loss": loss.item()}
            if configs["schedule"] != "none":
                log_dict["Current Learning Rate"] = get_current_learning_rate(optimizer)

            if configs["wandb"]:
                wandb.log(log_dict)
            else:
                print(log_dict)
        if configs["mixed_precision"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if configs["schedule"] == "cos":
            if "samples_per_epoch" in configs:
                num_steps = configs["samples_per_epoch"] // configs["batch_size"]
            else:
                num_steps = 10000
            adjust_learning_rate(optimizer, idx / num_steps + epoch, configs)


def train(configs):
    print("=" * 20)
    print("Initializing classification trainer")
    print("=" * 20)
    metrics = utils.initialize_metrics(configs)
    criterion = utils.create_loss(configs)
    base_model = model_utils.create_model(configs)
    optimizer = utils.create_optimizer(configs)(
        base_model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )

    # compile model (torch 2.0)
    # model = torch.compile(base_model)
    model = base_model
    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    train_loader, val_loader, _ = utils.create_dataloaders(configs)
    model.to(configs["device"])
    best_loss = 10000.0
    if configs["start_epoch"] is None:
        start_epoch = 0
    else:
        start_epoch = configs["start_epoch"]

    early_stop_counter = 0

    for epoch in range(start_epoch, configs["epochs"]):
        train_epoch(train_loader, model, optimizer, criterion, epoch, configs, scaler)
        val_loss = test(configs, phase="val", model=model, criterion=criterion, loader=val_loader, epoch=epoch)
        if "early_stopping" in configs and configs["early_stopping"] > 0:
            early_stop, early_stop_counter = early_stopping(
                val_loss, best_loss, early_stop_counter, patience=configs["early_stopping"]
            )
        else:
            early_stop = False
        if val_loss < best_loss:
            best_loss = val_loss
            print("New best validation loss: ", best_loss)
            print("Saving checkpoint")
            # Store checkpoint
            torch.save(base_model, os.path.join(configs["checkpoint_path"], "best_model.pt"))
        torch.save(
            base_model.state_dict(),
            os.path.join(configs["checkpoint_path"], "best_model_state_dict_" + str(epoch) + ".pt"),
        )
        if early_stop:
            print("Early stopping at epoch: ", epoch)
            break


def test(configs, phase, model=None, loader=None, criterion=None, epoch="Test"):
    if phase == "test":
        print("=" * 20)
        print("Begin Testing")
        print("=" * 20)
        _, _, loader = utils.create_dataloaders(configs)
        criterion = utils.create_loss(configs)

        # Load model from checkpoint
        model = torch.load(os.path.join(configs["checkpoint_path"], "best_model.pt"), map_location=configs["device"])

        # compile model
        model = torch.compile(model)

    elif phase == "val":
        print("=" * 20)
        print("Begin Evaluation")
        print("=" * 20)
    else:
        print("Uknown phase!")
        exit(3)

    if configs["linear_evaluation"] or configs["fully_finetune"]:
        modality_dictionary = {v: k for k, v in configs["modality_channels"].items()}
        available_modalities = configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        desired_indices = []
        for modality in available_modalities:
            spectral_keys.append(int(modality_dictionary[modality]))
            desired_indices.append(available_modalities[modality])

    if "polynet_evaluation" in configs and configs["polynet_evaluation"]:
        polynet_evaluation = True
        resnet_star_configs = json.load(open("configs/datasets/supervised_foundation_cls.json", "r"))
        modality_dictionary = {v: k for k, v in resnet_star_configs["modality_channels"].items()}
        available_modalities = resnet_star_configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        desired_indices = []
        for modality in available_modalities:
            spectral_keys.append(int(modality_dictionary[modality]))
            desired_indices.append(available_modalities[modality])
        print("Desired indices: ", desired_indices)
    else:
        polynet_evaluation = False

    metrics = utils.initialize_metrics(configs)
    model.to(configs["device"])
    model.eval()
    total_loss = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm.tqdm(loader)):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
                image, label = batch
                image = image.to(configs["device"])
                label = label.to(configs["device"])

                if not configs["linear_evaluation"] and not configs["fully_finetune"]:
                    if not polynet_evaluation:
                        out = model(image)
                    else:
                        image = image[:, desired_indices, :, :]
                        out = model(image)
                else:
                    image = image[:, desired_indices, :, :]
                    out = model((image, spectral_keys))

                loss = criterion(out, label)
                total_loss += loss.item()
                for metric in metrics:
                    if metric.__class__.__name__ == "MultilabelAveragePrecision":
                        label = label.int()
                    metric(out, label)
                num_samples += image.shape[0]

    total_loss = total_loss / num_samples
    log_dict = {"Epoch": epoch, phase + " loss": total_loss}

    for idx, metric in enumerate(metrics):
        if metric.__class__.__name__ == "MultilabelCoverageError":
            log_dict[phase + " " + metric.__class__.__name__] = metric.compute()
        elif metric.average != "none":
            log_dict[phase + " " + metric.average + " " + metric.__class__.__name__] = metric.compute()
        else:
            if phase != "val":
                scores = metric.compute()
                for idx in range(scores.shape[0]):
                    log_dict[phase + " " + metric.__class__.__name__ + " Class: " + str(idx)] = scores[idx]

    if configs["wandb"]:
        wandb.log(log_dict)
    else:
        print(log_dict)
    return total_loss
