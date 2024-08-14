import os
import pprint

import kornia
import numpy as np
import pyjson5 as json
import torch
import torch.nn as nn
import tqdm
import wandb
from torch.optim.lr_scheduler import ExponentialLR

import utilities.model_utilities as model_utils
import utilities.utils as utils
import utilities.yolov5.loss as yolov5_loss


def train_epoch(train_loader, model, optimizer, epoch, configs, scaler, iteration, scheduler=None):
    model.train()
    image_processor = utils.create_image_processor(configs)
    if configs["architecture"].lower() == "fomonet":
        available_modalities = configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        for modality in available_modalities:
            spectral_keys.append(available_modalities[modality])
    else:
        spectral_keys = None

    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
        iteration += 1
        inputs, targets = batch
        images = torch.stack(inputs).to(configs["device"])
        if configs["det_format"] == "pascal_voc":
            targets = [
                {
                    "boxes": target["boxes"].to(configs["device"]),
                    "labels": target["labels"].to(configs["device"]),
                    "image_id": target["image_id"].to(configs["device"]),
                    "area": target["area"].to(configs["device"]),
                    "iscrowd": target["iscrowd"].to(configs["device"]),
                }
                for target in targets
            ]
        elif configs["det_format"] == "yolo":
            # formatting targets with img/batch index + single tensor for all targets
            targets = [
                torch.hstack((torch.tensor([i] * targets[i].shape[0]).unsqueeze(1), targets[i]))
                for i in range(len(targets))
            ]
            targets = torch.vstack(targets).to(configs["device"])

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
            if configs["architecture"] == "yolov5":
                out = model(images)
                loss = yolov5_loss.ComputeLoss(model)(out, targets)[0]  # keep only global loss
            elif configs["architecture"] == "yolos":
                # Note that YOLOS works with the COCO format (as input)
                embeddings = image_processor(images)["pixel_values"]
                embeddings = torch.stack([torch.Tensor(embedding) for embedding in embeddings])
                embeddings = embeddings.to(configs["device"])
                # format boxes to yolo format, while keeping coco structure
                for target in targets:
                    boxes = [
                        torch.stack(
                            utils.format_bboxes_coco_to_yolo(
                                box, target_im_size=embeddings.shape[2:], source_im_size=images.shape[2:]
                            )
                        )
                        for box in target["annotations"]["bbox"]
                    ]
                    target["annotations"]["bbox"] = torch.stack(boxes)

                targets = [
                    {
                        "image_id": target["annotations"]["image_id"].to(configs["device"]),
                        "class_labels": target["annotations"]["category_id"].to(configs["device"]),
                        "boxes": target["annotations"]["bbox"].to(configs["device"]),
                        "area": target["annotations"]["area"].to(configs["device"]),
                        "size": torch.tensor(embeddings.shape[2:]).to(configs["device"]),
                        "orig_size": torch.tensor(images.shape[2:]).to(configs["device"]),
                    }
                    for target in targets
                ]
                out = model(pixel_values=embeddings, labels=targets)
                loss = out["loss"]
            else:
                if configs["architecture"].lower() == "fomonet":
                    # This is with custom faster rcnn
                    if "exceptional_resize" in configs.keys():
                        if isinstance(configs["exceptional_resize"], int):
                            images = nn.functional.interpolate(
                                images, (configs["exceptional_resize"], configs["exceptional_resize"])
                            )
                    out = model(images=images, spectral_keys=spectral_keys, targets=targets)
                    loss = sum(sub_loss for sub_loss in out.values())
                elif configs["backbone"] == "dinov2":
                    inputs = image_processor(images)
                    inputs = [torch.tensor(img).to(configs["device"]) for img in inputs["pixel_values"]]
                    out = model(images=inputs, spectral_keys=None, targets=targets)
                    loss = sum(sub_loss for sub_loss in out.values())
                else:
                    out = model(images, targets)
                    # Losses are computed in the model for train mode
                    loss = sum(sub_loss for sub_loss in out.values())

        if  configs["architecture"].lower() == "fomonet" or configs["linear_evaluation"]:
            model_utils.adjust_learning_rate(optimizer, iteration, configs)

        if iteration % 20 == 0:
            if  configs["architecture"].lower() == "fomonet" or configs["linear_evaluation"]:
                log_dict = {
                    "Epoch": epoch,
                    "Iteration": iteration,
                    "train loss": loss.detach().cpu().item(),
                    "lr": model_utils.get_current_learning_rate(optimizer),
                }
            else:
                log_dict = {
                    "Epoch": epoch,
                    "Iteration": iteration,
                    "train loss": loss.detach().cpu().item(),
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
            if configs["linear_evaluation"] != True:
                if iteration % configs["lr_step"] == 0:
                    scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            if configs["linear_evaluation"] != True:
                if iteration % configs["lr_step"] == 0:
                    scheduler.step()
    return iteration


def train(configs):
    print("=" * 20)
    print("Initializing detection trainer")
    print("=" * 20)
    metrics = utils.initialize_metrics(configs)
    # criterion = utils.create_loss(configs)
    model = model_utils.create_model(configs)
    print("=" * 20)
    print("Number of trainable parameters: {}".format(utils.count_params(model)))
    print("=" * 20)

    if "momentum" in configs.keys():
        optimizer = utils.create_optimizer(configs)(
            model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"], momentum=configs["momentum"]
        )
    else:
        optimizer = utils.create_optimizer(configs)(
            model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
        )

    if configs["linear_evaluation"]:
        scheduler = None
    else:
        # TODO: load scheduler and optimizer if using a checkpoint
        scheduler = ExponentialLR(optimizer, gamma=0.9)

    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # compile model (torch 2.0)
    # torch.compile(base_model)
    train_loader, val_loader, _ = utils.create_dataloaders(configs)
    model.to(configs["device"])
    best_score = 0.0
    iteration = 0
    for epoch in range(configs["epochs"]):
        iteration = train_epoch(train_loader, model, optimizer, epoch, configs, scaler, iteration, scheduler)
        val_score = test(configs, phase="val", model=model, loader=val_loader, epoch=epoch)
        if val_score > best_score:
            best_score = val_score
            print("New best validation score: ", best_score)
            print("Saving checkpoint")
            # Store checkpoint
            torch.save(model, os.path.join(configs["checkpoint_path"], "best_model.pt"))


def test(configs, phase, model=None, loader=None, epoch=None, dataset_name=None):
    if phase == "test":
        print("=" * 20)
        print("Begin Testing")
        print("=" * 20)
        if loader is None:
            _, _, loader = utils.create_dataloaders(configs)

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

    metrics = utils.initialize_metrics(configs)
    model.to(configs["device"])
    model.eval()
    image_processor = utils.create_image_processor(configs)
    iou_thresh = configs["iou_thresh"]
    conf_thresh = configs["conf_thresh"]
    if  configs["architecture"].lower() == "fomonet":
        available_modalities = configs["dataset_modality_index"][configs["dataset"]]
        spectral_keys = []
        for modality in available_modalities:
            spectral_keys.append(available_modalities[modality])
    else:
        spectral_keys = None

    # Images to log to wandb
    first_image = None
    first_prediction = None
    first_mask = None
    class_id_to_label = utils.create_class_id_to_label(configs["dataset"])
    num_samples = 0
    full_predictions = []
    full_targets = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(loader)):
            inputs, targets = batch
            images = torch.stack(inputs).to(configs["device"])

            if configs["det_format"] == "pascal_voc":
                targets = [
                    {
                        "boxes": target["boxes"].to(configs["device"]),
                        "labels": target["labels"].to(configs["device"]),
                        "image_id": target["image_id"].to(configs["device"]),
                        "area": target["area"].to(configs["device"]),
                        "iscrowd": target["iscrowd"].to(configs["device"]),
                    }
                    for target in targets
                ]

            elif configs["det_format"] == "yolo":
                # formatting targets with img/batch index + single tensor for all targets
                targets = [
                    torch.hstack((torch.tensor([i] * targets[i].shape[0]).unsqueeze(1), targets[i]))
                    for i in range(len(targets))
                ]
                targets = torch.vstack(targets).to(configs["device"])

            torch.cuda.synchronize()
            if configs["architecture"] == "yolov5":
                out = model(images)
                loss = yolov5_loss.ComputeLoss(model)(out[1], targets)[0]  # keep only global loss
                # formatting output/targets to match pipeline
                nb_imgs = len(images)
                nb_classes = configs["num_classes"]
                # outputs are in 'coco' mode, transform them back into pascal voc
                out = [
                    {
                        "boxes": utils.format_bboxes_coco_to_pascal(out_sample[:, :4]),
                        "scores": out_sample[:, 4 : 5 + nb_classes].softmax(-1).max(-1)[0],
                        "labels": out_sample[:, 4 : 5 + nb_classes].softmax(-1).max(-1)[1],
                    }
                    for out_sample in out[0]
                ]
                # Reformat target from yolo to pascal for the evaluation pipeline
                targets = [
                    {
                        "boxes": utils.format_bboxes_yolo_to_pascal(targets[targets[:, 0] == i][:, 2:], images.shape[2:]),
                        "labels": targets[targets[:, 0] == i][:, 1].int(),
                    }
                    for i in range(nb_imgs)
                ]
                # For evaluation only: nms + conf filtering
                out = [utils.apply_nms(out_frame, iou_thresh=iou_thresh) for out_frame in out]
                out = [utils.apply_conf(out_frame, conf_thresh=conf_thresh) for out_frame in out]

            elif configs["architecture"] == "yolos":
                # Note that YOLOS works with the COCO format (as input)
                embeddings = image_processor(images)["pixel_values"]
                embeddings = torch.stack([torch.Tensor(embedding) for embedding in embeddings])
                embeddings = embeddings.to(configs["device"])
                # Target stay in coco format here, we don't care about the loss
                targets = [
                    {
                        "image_id": target["annotations"]["image_id"].to(configs["device"]),
                        "class_labels": target["annotations"]["category_id"].to(configs["device"]),
                        "labels": target["annotations"]["category_id"].to(configs["device"]),
                        "boxes": target["annotations"]["bbox"].to(configs["device"]),
                        "area": target["annotations"]["area"].to(configs["device"]),
                        "size": torch.tensor(embeddings.shape[2:]).to(configs["device"]),
                        "orig_size": torch.tensor(images.shape[2:]).to(configs["device"]),
                    }
                    for target in targets
                ]
                out = model(pixel_values=embeddings, labels=targets)
                pred_boxes = out["pred_boxes"]
                pred_boxes = [utils.format_bboxes_yolo_to_pascal(pred_box, images.shape[2:]) for pred_box in pred_boxes]
                conf, pred_labels = out["logits"].softmax(-1).max(-1)
                out = [
                    {"boxes": pred_boxes[i], "scores": conf[i], "labels": pred_labels[i]} for i in range(images.shape[0])
                ]
                # For evaluation only: nms + conf filtering
                out = [utils.apply_nms(out_frame, iou_thresh=iou_thresh) for out_frame in out]
                out = [utils.apply_conf(out_frame, conf_thresh=conf_thresh) for out_frame in out]
                # Target boxes to pascal voc
                for i in range(len(targets)):
                    targets[i]["boxes"] = torch.stack(
                        [torch.tensor([box[0], box[1], box[0] + box[2], box[1] + box[3]]) for box in targets[i]["boxes"]]
                    ).to(configs["device"])
            elif configs["backbone"] == "dinov2":
                inputs = image_processor(images)
                inputs = [torch.tensor(img).to(configs["device"]) for img in inputs["pixel_values"]]
                out = model(images=inputs, spectral_keys=None)
                out = [utils.apply_nms(out_frame, iou_thresh=iou_thresh) for out_frame in out]
                out = [utils.apply_conf(out_frame, conf_thresh=conf_thresh) for out_frame in out]
            else:
                if configs["architecture"].lower() == "fomonet":
                    # This is with custom faster rcnn
                    out = model(images=images, spectral_keys=spectral_keys)
                    out = [utils.apply_nms(out_frame, iou_thresh=iou_thresh) for out_frame in out]
                    out = [utils.apply_conf(out_frame, conf_thresh=conf_thresh) for out_frame in out]
                else:
                    out = model(images)
                    # For evaluation only: nms + conf filtering
                    out = [utils.apply_nms(out_frame, iou_thresh=iou_thresh) for out_frame in out]
                    out = [utils.apply_conf(out_frame, conf_thresh=conf_thresh) for out_frame in out]

            if idx == 0:
                # Get a random image from the first batch
                rd_idx = np.random.randint(len(images))
                first_image = images[rd_idx].detach().cpu()
                first_prediction = out[rd_idx]
                first_target = targets[rd_idx]
            # for metric in metrics:
            #     metric.update(out, targets)
            full_predictions += out
            full_targets += targets
            num_samples += len(images)
            # torch.cuda.empty_cache()

    log_dict = {"Epoch": epoch}
    for idx, metric in enumerate(metrics):
        metric.update(full_predictions, full_targets)
        scores = metric.compute()
        for metric_name in scores.keys():
            if metric_name != "classes":
                if phase == "test" and dataset_name:
                    if scores[metric_name].numel() == 1:
                        log_dict[phase + " " + dataset_name + " " + metric_name] = scores[metric_name].item()
                    else:
                        for i in range(len(scores[metric_name])):
                            log_dict[phase + " " + dataset_name + " " + metric_name + "/cls_" + str(i)] = scores[
                                metric_name
                            ][i].item()
                else:
                    if scores[metric_name].numel() == 1:
                        log_dict[phase + " " + metric_name] = scores[metric_name].item()
                    else:
                        for i in range(len(scores[metric_name])):
                            log_dict[phase + " " + metric_name + "/cls_" + str(i)] = scores[metric_name][i].item()
            if metric_name == configs["eval_metric"]:
                evaluation_score = scores[metric_name].item()

    if configs["wandb"]:
        # Store image every 20 epochs:
        if epoch%20 == 0:
            if configs["normalization"] == "standard":
                first_image = first_image.unsqueeze(0)
                first_image = kornia.enhance.Denormalize(torch.tensor(configs["mean"]), torch.tensor(configs["std"]))(
                    first_image
                ).squeeze()
                first_image = first_image[:3, :, :].permute(1, 2, 0) / first_image.max()
                first_image *= 255
            elif configs["normalization"] == "none":
                first_image = first_image[:3, :, :].permute(1, 2, 0)
            else:
                first_image = first_image[:3, :, :].permute(1, 2, 0) * 255

            im_size = list(first_image.size())[:2]
            # Everything should be in Pascal format before
            first_prediction_formatted = utils.format_bboxes_wandb(first_prediction, "pascal_voc", im_size)
            first_target_formatted = utils.format_bboxes_wandb(first_target, "pascal_voc", im_size)
            boxes_img_gt = wandb.Image(
                (first_image).int().cpu().detach().numpy(),
                boxes={"ground_truth": {"box_data": first_target_formatted, "class_labels": class_id_to_label}},
            )
            boxes_img_pred = wandb.Image(
                (first_image).int().cpu().detach().numpy(),
                boxes={"predictions": {"box_data": first_prediction_formatted, "class_labels": class_id_to_label}},
            )
            log_dict[phase + " sample GT"] = boxes_img_gt
            log_dict[phase + " sample predictions"] = boxes_img_pred
        # Still pushing the logs at each epoch
        wandb.log(log_dict)
    else:
        print(log_dict)

    return evaluation_score
