### This repository contains the code used in [FoMo: Multi-Modal, Multi-Scale and Multi-Task Remote Sensing Foundation Models for Forest Monitoring](https://arxiv.org/abs/2312.10114) (WIP)


If you use this work please consider citing:

```
@article{bountos2023fomo,
  title={FoMo-Bench: a multi-modal, multi-scale and multi-task Forest Monitoring Benchmark for remote sensing foundation models},
  author={Bountos, Nikolaos Ioannis and Ouaknine, Arthur and Rolnick, David},
  journal={arXiv preprint arXiv:2312.10114},
  year={2023}
}
```
# Table of Contents
- [Setup Project](#setup-project)
- [Repository Structure](#repository-structure)
- [Downloading the data](#downloading-datasets)
- [TalloS](#tallos)
- [Running Experiments](#experiments)
    - [Supported Models](#supported-models)
    - [Adding new models](#adding-new-models)
    - [Adding new tasks](#adding-new-tasks)
    - [Training Foundation Models](#train-foundation-models)
- [Add new augmentation methods](#adding-data-augmentations)
- [Add new datasets](#adding-new-datasets)
- [Webdataset setup](#webdataset-setup)
- [Supported datasets](#datasets-in-the-benchmark)
- [Loading pretrained FoMo-Net](#loading-pretrained-fomo-net)
- [Prtrained supervised baselines](#pretrained-supervised-baselines)

### Setup project

This code has been tested with python 3.10. To install all necessary packages run:

```
pip install -r requirements.txt
```

Depending on the cuda version in your system execute (e.g for cuda 12.1):
```
pip install pyg_lib torch_scatter torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

To activate the pre-commit hook for the [black formatter](https://black.readthedocs.io/en/stable/) execute:
```
pre-commit install
```

### Repository Structure
```css
.
├── configs/
│   ├── configs.json
│   ├── datasets
│   ├── method
│   ├── download
│   └── training
├── datasets/
│   ├── BigEarthNet.py
│   ├── FLAIRDataset.py
│   └──  ...
├── downloading_scripts/
|    ├── bigearthnet.sh
|    ├── treesat.sh
|    └── ...
├── training/
|    └── classification.py
|    └── segmentation.py
└── utilities/
     └── augmentations.py
     └── utils.py
     └── model_utilities.py
     └── webdataset_writer.py
main.py
downloader.py
```

`configs.json` contains high-level experimental choices, including the dataset of interest and whether to activate wandb.


the `datasets/` directory contains the dataset specific configurations e.g task to solve, metrics to log etc. 

Example configurations for the `cactus` dataset:

```json5
{
    "root_path":"dataset_root_path",
    "task":"classification", // Possible Tasks: Depending on the dataset,
    "metrics": ["accuracy","fscore"], //Desired metrics to log
    "num_classes":2,
    "in_channels":3,
    "meta_info":""
}
```

Similarly, the method and training directories contain the configuration choices for the desired tasks e.g classification, segmentation and training choices e.g batch size, epochs etc respectively.


### Downloading Datasets

To download the datasets used in this benchmark select the desired dataset in `configs/download/download.json` along with the base directory to store the data and execute

 `python downloader.py`. 
 
 Make sure to give proper permissions to the scripts under `downloading_scripts/` running:
 
  `chmod +x download_script.sh`

`downloader.py` will handle the downloading and all necessary restructuring needed for the experiments.


**Note for object detection and point cloud datasets**: we provide scripts to create tiles or sub-point clouds for the NeonTree, ReforesTree and FORinstance datasets. Please refer to the corresponding scripts in the utilities folder, either for [detection](utilities/detection_datasets) or [point cloud](utilities/pointcloud_datasets) datasets.    


### TalloS

TalloS Dataset can be found in the following [dropbox folder](https://www.dropbox.com/scl/fo/4h6dihx22gjg9vl8ofiek/AAm_Dwscf7u1mhQS0tqDQSY?rlkey=cpcal7kmrj55yihxvu3xkmshs&st=99b4yytx&dl=0%5D(https://www.dropbox.com/scl/fo/4h6dihx22gjg9vl8ofiek/AAm_Dwscf7u1mhQS0tqDQSY?rlkey=cpcal7kmrj55yihxvu3xkmshs&st=8irk22zn&dl=0)).

### Experiments

All information is aggregated by `main.py`. Given the aggregated configurations, the appropriate dataloading, training and testing functions are constructed.

 For each available task, the data loader follows same/similar patterns, enabling the training/testing procedures to remain (mostly) dataset agnostic.

 To run an experiment one has to select the desired dataset in `configs/configs.json`. The training options can be defined in `configs/training/training.json`. If a dataset can support multiple tasks, the user can specify the desired task in `configs/datasets/[YOUR_DATASET].json`.

 All datasets support the [webdataset](https://github.com/webdataset/webdataset) format for more efficient data loading. To enable it set `webdataset:true` in `configs/configs.json`. If webdataset shards exist, training begins immediately. Otherwise, the shards are created automatically. Depending on the dataset size this process may take from minutes to a few hours.

If data augmentation is needed then set `augment:true` in `configs/training/training.json`. The desired data augmentations can be set in `configs/augmentations/augmentations.json`, along with their strength (probability of occuring) e.g if we want to always resize an image to 224x224 pixels set: 
```json5
"Resize":{
            "value":224,
            "p":1.0
        },
```
The current augmentation configuration files contains all supported augmentations.

#### Webdataset setup
The example in `configs/configs.json` contains the following options for webdataset:
```
"webdataset":true,
"webdataset_shuffle_size": 1000,
"webdataset_initial_buffer":1000,
"max_samples_per_shard": 256, //set upper limit 256 samples per shard
"webdataset_root_path": null,
```
Setting the `webdtaset_root_path` variable will change the saving directory of the webdataset. If left at null, the webdataset will be saved at the same directory as the dataset of interest. The `max_samples_per_shard` argument is only used when creating the webdataset and refers to the maximum number of samples that would be contained in a single shard. This is handled by `utilities/webdataset_writer.py`. `webdataset_shuffle_size` determines the size of the buffer where the data are sampled, while `webdataset_initial_buffer` the amount of samples to be loaded before starting to yield. 

#### Supported models

For classification tasks we support all encoders available in [timm](https://github.com/huggingface/pytorch-image-models). 
In this benchmark we mainly focus on:

| Model | Paper | 
| :---: | :---: |
|ResNet| [ResNet Paper](https://arxiv.org/abs/1512.03385)|
|ViT| [ViT Paper](https://arxiv.org/abs/2010.11929)|
|ConvNext|[ConvNext Paper](https://arxiv.org/abs/2201.03545)|

In terms of semantic semgnetation problems we focus on:

| Model | Paper | 
| :---: | :---: |
|UNet| [UNet Paper](https://arxiv.org/abs/1505.04597)|
|UNet++|[UNet++ Paper](https://arxiv.org/abs/1807.10165)|
|DeepLabv3plus|[DeepLabv3plus Paper](https://arxiv.org/abs/1802.02611)|
|UperNet|[UperNet Paper](https://arxiv.org/abs/1807.10221)|


For object detection we support:

|Model| Paper|
|Faster R-CNN| [Faster R-CNN paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)|
|RetinaNet| [RetinaNet paper](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)|
|YOLOS| [YOLOS paper](https://proceedings.neurips.cc/paper/2021/hash/dc912a253d1e9ba40e2c597ed2376640-Abstract.html)|

#### Adding new models
To add support for a new model just include its construction in `utilities/model_utilities.py`. Depending on the task, the models are constructed in one of the following functions:
```
create_classifier() // for classification tasks,
create_segmentor() // for semantic segmentation tasks and,
create_detector() // for object detection tasks
```
#### Adding new tasks
In case one needs to solve a task not included in this repo, they have to update the pipeline with the following steps:

1.  Create a training/testing procedure in `training/` as done in `training/classification.py`.
2. Update the `create_procedures()` function in the `utilities/utils.py`, to handle the desired task e.g for classification:
```
if configs['task']=='classification':
    trainer = classification.train
    tester = classification.test
```
3. Create a config file specifying the desired model and other hyperparameters needed for the training/testing procedures in `configs/method/your_task.json`. Examples can be found in `configs/method/` e.g `configs/method/classification.json`.

4. Update the `create_checkpoint_path()` function in `utilities/utils.py` to create a unique checkpoint for the given task given the provided configs for each experiment. E.g for the semantic segmentation task:
```
 if configs['task']=='segmentation':
        checkpoint_path = (
                Path("checkpoints")
                / configs["task"].lower()
                / configs["dataset"].lower()
                / configs["architecture"].lower()
                / configs["backbone"].lower()
            )
```

5. Run your experiments by modifying your task config file `configs/method/your_task.json` and running `python main.py`.


#### Train FoMo-Net

To enable FoMo-Net training set "all" as dataset in `configs.json`. 
`configs/datasets/all.json` provides an example of the needed configurations. Set augmentations to true in `configs/training/training.json` and the desired augmentations in `configs/augmentations/augmentations.json` e.g
```
"RandomResizedCrop": {
            "value": 224,
            "scale":[0.2, 1.0],
            "interpolation":3,
            "p": 1.0
        },
```
### Adding data augmentations
The data augmentation pipeline is based on the [Albumentations](https://albumentations.ai/docs/) library. To add an augmentation method one should include it in the `get_augmentations` function of `utilities/augmentations.py`. For example to add the `VerticalFlip` augmentation we add:

```python
elif k == "VerticalFlip":
    aug = A.augmentations.VerticalFlip(p=v["p"])
```


 ### Adding new datasets

 To add a new dataset one has to:
    
    1. Create a configuration file in configs/datasets/ with the name of the dataset in lower case (e.g configs/datasets/flair.json)
        - The configuration file has to include:
            - The root path of the data
            - The nature of the task to solve (e.g classification)
            - The metrics to log in a list (e.g ["accuracy","fscore"])
            - Any other information needed for loading the data (depends on your implementation of the data loader).
    2. Create a data loader in datasets/ (e.g datasets/FLAIRDataset.py). Each dataset should include a plot() function for visualization.
    3. The data loader should return data in the following form: `sample, label`. If a different scheme is used, the training procedures should be adapted accordingly (e.g classification.py and the webdataset processing pipeline.)
    4. Include the option to load the dataset in the load_dataset() function of the utilities/utils.py
        - The following code block shows an example for the FLAIR dataset
    5. Include the mean and std of the new dataset to configs/stats/stats.json. These stats can be calculated using the calc_stats.py script. Set the option batched=True to process the dataset in batches.
```python
elif configs['dataset'].lower()=='flair':
    dataset = datasets.FLAIRDataset.FLAIRDataset(configs,mode)
```

### Datasets in the benchmark
The following table presents the datasets supported in this repo, along with some basic information regarding the data sensor, their spatial coverage and the tasks they enable. Each dataset can be used in the respective config files with the following names in lower case.


| Dataset | Modalities | Possible Tasks | Covered Areas | 
| :---: | :---: | :---: | :---: |
| [Cactus](https://www.sciencedirect.com/science/article/abs/pii/S1574954119300895) | Aerial RGB |Classification | Mexico|
| [FLAIR](https://arxiv.org/pdf/2211.12979.pdf) | Aerial - RGB, NIR, Elevation|Segmentation| France|
| [FLAIR2](https://arxiv.org/pdf/2305.14467.pdf) | Aerial - RGB, NIR, Elevation, Sentinel-2| France|
| [TreeSatAI](https://essd.copernicus.org/articles/15/681/2023/)|Aerial, Sentinel-1, Sentinel-2 | Classification |Germany|
| [Woody](https://www.sciencedirect.com/science/article/abs/pii/S0034425719301166)| Aerial | Segmentation | Chile|
| [ReforesTree](https://ojs.aaai.org/index.php/AAAI/article/view/21471)| Aerial |Detection, Regression | Ecuador|
| [ForestNet](https://stanfordmlgroup.github.io/projects/forestnet/)| Landsat-8|Classification, Segmentation| Indonesia|
| [NeonTree](https://zenodo.org/record/5914554#.YfRhcPXMKHE)| Satellite RGB, LiDAR, Hyperspectral| Detection |USA|
| [Spekboom](https://zenodo.org/record/7564954)| AERIAL| Segmentation| South Africa|
| [Waititu](https://zenodo.org/record/7648984#.Y_gBqLTMKBQ)| Aerial | Segmentation| New Zealand|
| [BigEarthNet-MM](https://arxiv.org/pdf/2105.07921.pdf) | Sentinel-1, Sentinel-2| Multi-label Classification | Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, Switzerland|
| [Sen12MS](https://arxiv.org/pdf/1906.07789.pdf)|Sentinel-1, Sentinel-2|Multi-label Classification| Global|
| [RapidAI4EO](https://rapidai4eo.source.coop/)|Planet, Sentinel-2|Multi-label Classification| Europe|
| [TalloS]()|Sentinel-1, Sentinel-2, DEM, ERA-5|Multi-label Classification| Global|


### Loading pretrained FoMo-Net
The pretrained weights for FoMo-Net$`_1`$ can be accessed [here](https://www.dropbox.com/scl/fi/4ckmxlcbc0tcod8hknp7c/fomo_single_embedding_layer_weights.pt?rlkey=26tlf3yaz93vvcosr0qrvklub&st=lm17tghn&dl=0).

`examples/pretrained_fomo_example.py` presents a minimal example of initializing FoMo-Net$`_1`$ and loading the pretrained weights.

### Pretrained supervised baselines
 FoMo-Bench baseline checkpoints trained with a random seed set to 222:
  - TalloS:
      - [ConvNeXt](https://www.dropbox.com/scl/fi/awflxfi5kwv803fu28se4/convnext_tallos.pt?rlkey=euyjn43b870hca75lgisztbus&st=nnggodxa&dl=0)
  - TreeSatAI (Sentinel-2 200m):
      - [ConvNeXt](https://www.dropbox.com/scl/fi/jqtj98icjed5yh9gr1vs6/convnext_treesat.pt?rlkey=v5cvt9x4ijt0pucjjf00eaxs2&st=uai2s6zz&dl=0)
   
 `examples/pretrained_fomobench_example.py` presents a minimal example of loading the pretrained weights.
