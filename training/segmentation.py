import os
import pprint

import kornia
import numpy as np
import pyjson5 as json
import torch
import tqdm

import utilities.model_utilities as model_utils
import utilities.utils as utils
import wandb
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
                out = model(image)
            else:
                image = image[:, desired_indices, :, :]

                out = model((image, spectral_keys))

            label = label.long()
            loss = criterion(out, label)
        if idx % 100 == 0:
            log_dict = {"Epoch": epoch, "Iteration": idx, "train loss": loss.item()}
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
    print("Initializing segmentation trainer")
    print("=" * 20)
    metrics = utils.initialize_metrics(configs)
    criterion = utils.create_loss(configs)
    base_model = model_utils.create_model(configs)
    optimizer = utils.create_optimizer(configs)(
        base_model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )

    # compile model (torch 2.0)
    model = base_model  # torch.compile(base_model)

    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    train_loader, val_loader, _ = utils.create_dataloaders(configs)
    model.to(configs["device"])
    best_loss = 10000.0
    early_stop_counter = 0
    for epoch in range(configs["epochs"]):
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

        if configs["eval_checkpoint"] is not None:
            print("=" * 20)
            print("Evaluating segmentation for phase: ", phase, " with checkpoint: ")
            print(configs["eval_checkpoint"])
            print("=" * 20)
            # Load model from checkpoint
            model = torch.load(os.path.join(configs["eval_checkpoint"]), map_location=configs["device"])
        else:
            # infer checkpoint path from configs
            # Load model from checkpoint
            model = torch.load(os.path.join(configs["checkpoint_path"], "best_model.pt"), map_location=configs["device"])

        # compile model (torch 2.0)
        # model = torch.compile(model)
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

    metrics = utils.initialize_metrics(configs)
    model.to(configs["device"])
    model.eval()
    total_loss = 0.0

    # Images to log to wandb
    first_image = None
    first_prediction = None
    first_mask = None
    num_samples = 0
    for idx, batch in enumerate(tqdm.tqdm(loader)):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
                image, label = batch
                image = image.to(configs["device"])
                label = label.to(configs["device"])

                if not configs["linear_evaluation"] and not configs["fully_finetune"]:
                    out = model(image)
                else:
                    image = image[:, desired_indices, :, :]
                    out = model((image, spectral_keys))
                if idx == 0:
                    predictions = out.argmax(1)
                    first_image = image.detach().cpu()[0]
                    first_prediction = predictions.detach().cpu()[0]
                    first_mask = label.detach().cpu()[0]
                label = label.long()
                loss = criterion(out, label)
                total_loss += loss.item()
                for metric in metrics:
                    metric(out, label)
                num_samples += image.shape[0]

    total_loss = total_loss / num_samples
    log_dict = {"Epoch": epoch, phase + " loss": total_loss}
    for idx, metric in enumerate(metrics):
        if metric.average != "none":
            log_dict[phase + " " + metric.average + " " + metric.__class__.__name__] = metric.compute()
        else:
            if phase != "val":
                scores = metric.compute()
                for idx in range(scores.shape[0]):
                    log_dict[phase + " " + metric.__class__.__name__ + " Class: " + str(idx)] = scores[idx]
    if configs["wandb"]:
        class_labels = {}
        for i in list(range(configs["num_classes"])):
            class_labels[i] = str(i)

        if configs["log_images"]:
            if configs["normalization"] == "standard":
                first_image = first_image.unsqueeze(0)
                first_image = kornia.enhance.Denormalize(torch.tensor(configs["mean"]), torch.tensor(configs["std"]))(
                    first_image
                ).squeeze()
                first_image = first_image[:3, :, :].permute(1, 2, 0) / first_image.max()
                first_image *= 255
            else:
                first_image = first_image[:3, :, :].permute(1, 2, 0) * 255
            mask_img = wandb.Image(
                (first_image).int().cpu().detach().numpy(),
                masks={
                    "predictions": {"mask_data": first_prediction.float().numpy(), "class_labels": class_labels},
                    "ground_truth": {"mask_data": first_mask.float().numpy(), "class_labels": class_labels},
                },
            )
            log_dict[phase + " sample"] = mask_img
        wandb.log(log_dict)
    else:
        print(log_dict)
    return total_loss
