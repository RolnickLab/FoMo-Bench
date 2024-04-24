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
from torch.optim.lr_scheduler import ExponentialLR


def train_epoch(train_loader, model, optimizer, criterion, epoch, configs, scaler, iteration, scheduler=None):
    if not configs["linear_evaluation"]:
        model.train()

    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
        iteration += 1
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
            batch_data = batch[0]
            point_cloud = [data.pos for data in batch_data]
            point_cloud = torch.vstack(point_cloud)
            # Empty features for benchmarking
            x = [data.x for data in batch_data]
            x = torch.vstack(x)
            label = [data.y for data in batch_data]
            label = torch.cat(label).to(configs["device"])
            point_cloud = point_cloud.to(configs["device"]).float()
            x = x.to(configs["device"]).float()
            label = label.to(configs["device"])
            if configs["architecture"] in ("pointnet2", "point_transformer"):
                batch = x.new_zeros(point_cloud.shape[0], dtype=torch.int64).to(configs["device"])
                out = model(x, point_cloud, batch)
            else:
                out = model(point_cloud)

            loss = criterion(out, label)

        if iteration % 10 == 0:
            log_dict = {
                "Epoch": epoch,
                "Iteration": iteration,
                "train loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
            }
            if configs["wandb"]:
                wandb.log(log_dict)
            else:
                print(log_dict)
        if configs["mixed_precision"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if iteration % configs["lr_step"] == 0:
                scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            if iteration % configs["lr_step"] == 0:
                scheduler.step()
    return iteration


def train(configs):
    print("=" * 20)
    print("Initializing segmentation trainer")
    print("=" * 20)
    metrics = utils.initialize_metrics(configs)
    criterion = utils.create_loss(configs)
    model = model_utils.create_model(configs)
    optimizer = utils.create_optimizer(configs)(model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"])
    print("Number of trainable parameters: {}".format(utils.count_params(model)))
    print("=" * 20)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    # compile model (torch 2.0)
    # model = torch.compile(model)

    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    train_loader, val_loader, _ = utils.create_dataloaders(configs)
    model.to(configs["device"])
    best_loss = 10000.0
    iteration = 0
    for epoch in range(configs["epochs"]):
        iteration = train_epoch(train_loader, model, optimizer, criterion, epoch, configs, scaler, iteration, scheduler)
        if (epoch + 1) % configs["val_step"] == 0:
            val_loss = test(configs, phase="val", model=model, criterion=criterion, loader=val_loader, epoch=epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                print("New best validation loss: ", best_loss)
                print("Saving checkpoint")
                # Store checkpoint
                torch.save(model, os.path.join(configs["checkpoint_path"], "best_model.pt"))


def test(configs, phase, model=None, loader=None, criterion=None, epoch="Test"):
    if phase == "test":
        print("=" * 20)
        print("Begin Testing")
        print("=" * 20)
        _, _, loader = utils.create_dataloaders(configs)
        criterion = utils.create_loss(configs)

        # Load model from checkpoint
        model = torch.load(os.path.join(configs["checkpoint_path"], "best_model.pt"), map_location=configs["device"])
        print("Number of trainable parameters: {}".format(utils.count_params(model)))
        print("=" * 20)

        # compile model (torch 2.0)
        # model = torch.compile(model)
    elif phase == "val":
        print("=" * 20)
        print("Begin Evaluation")
        print("=" * 20)
    else:
        print("Uknown phase!")
        exit(3)

    metrics = utils.initialize_metrics(configs)
    for metric in metrics:
        metric = metric.to("cpu")
    model.to(configs["device"])
    model.eval()
    total_loss = 0.0

    # Images to log to wandb
    first_image = None
    first_prediction = None
    first_mask = None
    num_samples = 0
    random_batch_idx = np.random.randint(len(loader))
    for idx, batch in enumerate(tqdm.tqdm(loader)):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
                batch_data = batch[0]
                point_cloud = [data.pos for data in batch_data]
                if idx == random_batch_idx:
                    # for visualization purposes
                    first_pc_len = point_cloud[0].shape[0]
                point_cloud = torch.vstack(point_cloud)
                # Empty features for benchmarking
                x = [data.x for data in batch_data]
                x = torch.vstack(x)
                label = [data.y for data in batch_data]
                label = torch.cat(label)
                point_cloud = point_cloud.to(configs["device"]).float()
                x = x.to(configs["device"]).float()
                label = label.to(configs["device"])
                if configs["architecture"] in ("pointnet2", "point_transformer"):
                    batch = x.new_zeros(point_cloud.shape[0], dtype=torch.int64).to(configs["device"])
                    out = model(x, point_cloud, batch)
                else:
                    out = model(point_cloud)

                loss = criterion(out, label)
                total_loss += loss.item()
                if idx == random_batch_idx:
                    # for visualization purposes
                    predictions = out.argmax(1)
                    first_pc = point_cloud.detach().cpu()[:first_pc_len].numpy()
                    first_pc_prediction = predictions.detach().cpu()[:first_pc_len].numpy()
                    first_pc_label = label.detach().cpu()[:first_pc_len].numpy()
                for metric in metrics:
                    metric(out.detach().cpu(), label.detach().cpu())
                num_samples += 1  # the entire PC is considered as a single sample for loss computation

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

        first_pc_prediction += 1
        first_pc_label += 1
        pc_vis = {
            "predictions": wandb.Object3D(np.hstack((first_pc, first_pc_prediction.reshape(-1, 1)))),
            "labels": wandb.Object3D(np.hstack((first_pc, first_pc_label.reshape(-1, 1)))),
        }

        log_dict[phase + " sample"] = pc_vis
        wandb.log(log_dict)
    else:
        print(log_dict)
    return total_loss
