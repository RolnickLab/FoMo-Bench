import os
import pickle
import pprint
import random
import warnings

from pathlib import Path
import cv2 as cv
import einops
import kornia
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjson5 as json
import rasterio
import torch
import xmltodict
from torchvision import transforms
from tqdm import tqdm

import albumentations as A
from albumentations.augmentations import Resize
from torch_geometric.data import Data
import torch_geometric.transforms as T

import utilities.augmentations
from utilities.utils import format_bboxes_voc_to_yolo

'''
Data loading for the NeonTree Dataset published in:
https://zenodo.org/record/5914554#.YfRhcPXMKHE
'''

class NeonTreeDataset(torch.utils.data.Dataset):

    def __init__(self, configs, mode='train'):
        self.configs = configs
        self.root_path = Path(configs['root_path'])
        self.mode = mode
        self.modality = self.configs['modality']
        if 'det_format' in self.configs.keys():
            self.det_format = self.configs['det_format']
        if 'nb_points' in self.configs.keys():
            self.nb_points = self.configs['nb_points']
        if self.configs['augment']:
            self.augmentations = utilities.augmentations.get_augmentations(configs)
        else:
            self.augmentations = None

        if self.mode == 'train' or self.mode == 'val':
            folder = 'training'
        elif self.mode == 'test':
            folder = 'evaluation'

        self.rgb_path = self.root_path / folder / 'RGB'
        self.lidar_path = self.root_path / folder / 'LiDAR'
        self.hyperspectral_path = self.root_path / folder / 'Hyperspectral'
        self.chm_path = self.root_path / folder / 'CHM'

        if self.modality.lower() == 'rgb':
            data_path = self.rgb_path
            if self.configs['normalization'] == 'standard':
                self.normalization = transforms.Normalize(mean=self.configs['mean'],std=self.configs['std'])

        elif self.modality.lower() == 'lidar':
            data_path = self.lidar_path
            self.normalization = T.NormalizeScale()

        elif self.modality.lower() == 'chm':
            data_path = self.chm_path

        else:
            print('Modality not supported. Exiting!')

        self.samples = []
        path_files = data_path.glob('*.pkl')
        for path_file in path_files:
            with open(path_file, 'rb') as f:
                self.samples.append(pickle.load(f))
        self.samples = [item for sublist in self.samples for item in sublist]

        if self.modality.lower() == 'rgb':
            self.max_boxes = 0
            for sample in self.samples:
                with open(sample['labels'], 'rb') as f:
                    bboxes = pickle.load(f)
                sample['bboxes'] = bboxes
                sample['num_boxes'] = len(bboxes)
                if len(bboxes)>self.max_boxes:
                    self.max_boxes = len(bboxes)

            # filter to keep images with boxes
            # Required for all splits for COCO API
            self.samples = [sample for sample in self.samples if len(sample['bboxes'])>0]
        elif self.modality.lower() == 'lidar':
            pass
        else:
            raise Exception('Modality {} is not supported yet.'.format(self.modality))

        # For debugging
        # self.samples = self.samples[:100]

        if self.mode=='train':
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[:int(0.9*len(self.samples))]

        elif self.mode=='val':
            random.Random(999).shuffle(self.samples)
            self.samples = self.samples[int(0.9*len(self.samples)):]

        self.num_examples = len(self.samples)
        print('Number of samples in split {} with modality: {}  = {}'.format(self.mode, self.modality, self.num_examples))

    def __len__(self):
        return self.num_examples

    def plot(self, index=0):
        temp_path = Paths().get()['temp']
        sample = self.samples[index]
        if self.modality=='rgb':
            with rasterio.open(sample['image']) as rgb_file:
                data = rgb_file.read()
            bbox = sample['bboxes']
            data = einops.rearrange(data,'c h w -> h w c')
            # Create figure and axis objects
            _, ax = plt.subplots(nrows=1, ncols=2)
            # Plot the image
            ax[0].imshow(data)
            ax[1].imshow(data)
            # Add bounding boxes to the plot
            if bbox:
                for box in bbox:
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min
                    rect = plt.Rectangle((x_min, y_min), w, h, fill=False, color='red')
                    ax[1].add_patch(rect)
            file_name = 'NeonTree_sample_' + str(index).zfill(5) + '.png'
            plt.savefig(temp_path / file_name)
            # plt.show()
            plt.close()

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.modality=='rgb':
            with rasterio.open(sample['image']) as rgb_file:
                image = rgb_file.read()
            bboxes = sample['bboxes']
            class_labels = ['tree']*(len(bboxes))
            if self.det_format == 'coco':
                # [x_min, y_min, x_max, y_max] -> [x_min, y_min, w, h]
                bboxes = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in bboxes]
            elif self.det_format == 'yolo':
                # [x_min, y_min, x_max, y_max] -> [x_min, x_max, y_min, y_max] -> [x_center, y_center, w, h] in relative coords
                img_size = image.shape[1:]
                bboxes = [format_bboxes_voc_to_yolo([box[0], box[2], box[1], box[3]], img_size)
                          for box in bboxes]

            if self.configs['augment'] and self.mode == "train":
                image = einops.rearrange(image, 'c h w -> h w c')
                transform = self.augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
                image = transform['image']
                bboxes = transform['bboxes']
                class_labels = transform['class_labels']
                image = einops.rearrange(image, 'h w c -> c h w')

            if self.mode in ('val', 'test') and 'Resize' in list(self.configs['augmentations'].keys()):
                image = einops.rearrange(image, 'c h w -> h w c')
                size = self.configs['augmentations']['Resize']['value']
                resizer = A.Compose([Resize(height=size, width=size, p=1.)],
                                    bbox_params=A.BboxParams(format=self.det_format, min_visibility=0.01,
                                                             label_fields=['class_labels']))
                transform = resizer(image=image, bboxes=bboxes, class_labels=class_labels)
                image = transform['image']
                bboxes = transform['bboxes']
                class_labels = transform['class_labels']
                image = einops.rearrange(image, 'h w c -> c h w')

            image = torch.from_numpy(image).float()
            if self.configs['normalization']=='minmax':
                image /= image.max()
            elif self.configs['normalization']=='standard':
                image = self.normalization(image)
            elif self.configs['normalization']=='none':
                pass    
            else:
                image /= 255.

            bboxes = np.array(bboxes)
            if self.det_format == 'coco':
                areas = bboxes[:, 2]*bboxes[:, 3]
            elif self.det_format == 'yolo':
                # not required
                pass
            else:
                # pascal voc format
                areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            labels = torch.ones(len(bboxes), dtype=torch.int64)
            iscrowd = torch.zeros(len(bboxes), dtype=torch.int64)

            target = dict()
            if self.det_format == 'coco':
                ann_info = {'id': torch.tensor(index),
                            'image_id': torch.tensor(index),
                            'category_id': labels,
                            'iscrowd': iscrowd,
                            'area': torch.tensor(areas),
                            'bbox': torch.tensor(bboxes).float(),
                            'segmentation': None}
                target['annotations'] = ann_info
                target['image_id'] = torch.tensor(index)
            elif self.det_format == 'yolo':
                bboxes = torch.tensor(bboxes)
                target = torch.hstack((labels.unsqueeze(1), bboxes))
            else:
                # pascal voc format
                target = {'image_id': torch.tensor(index),
                          'boxes': torch.tensor(bboxes).float(),
                          'area': torch.tensor(areas),
                          'iscrowd': iscrowd,
                          'labels': labels,
                          'num_boxes': sample['num_boxes']}
            return image, target

        elif self.modality=='chm':
            with rasterio.open(sample['chm']) as chm_file:
                data = chm_file.read()
            return data
        elif self.modality=='lidar':

            with laspy.open(sample['sub_pc']) as pc_file:
                data = pc_file.read()
            try:
                point_cloud = np.vstack([data.x, data.y, data.z, data.red, data.green, data.blue]).T
                # normalize colors
                point_cloud[:, 3:] /= 65280.0
            except AttributeError:
                point_cloud = np.vstack([data.x, data.y, data.z, data.intensity]).T
                # normalize intensity
                point_cloud[:, 3] /= 3780.0

            """
            point_cloud[:, 0] = (point_cloud[:, 0] - point_cloud[:, 0].min()) /\
                                (point_cloud[:, 0].max() - point_cloud[:, 0].min())
            point_cloud[:, 1] = (point_cloud[:, 1] - point_cloud[:, 1].min()) /\
                                (point_cloud[:, 1].max() - point_cloud[:, 1].min())
            point_cloud[:, 2] = (point_cloud[:, 2] - point_cloud[:, 2].min()) /\
                                (point_cloud[:, 2].max() - point_cloud[:, 2].min())
            """
            if self.configs['segmentation_task'] == 'binary_segmentation':
                labels = data.instance_id.copy()
                labels[labels != 0] = 1
                labels = np.array(labels, dtype=np.int64)
            elif self.configs['segmentation_task'] == 'instance_segmentation':
                labels = np.array(data.instance_id.copy(), dtype=np.int64)
            else:
                raise Exception('Segmentation format {} is not supported yet!'.format(self.seg_format))
            # Only point location are considered for the moment
            point_cloud = point_cloud[:, :3]
            # get random sub sample
            if self.mode == 'train':
                rand_idx = np.random.choice(list(range(point_cloud.shape[0])),
                                            size=self.nb_points,
                                            replace=False)
                point_cloud = point_cloud[rand_idx, :]
                labels = labels[rand_idx]
            
            point_cloud = torch.tensor(point_cloud)
            labels = torch.tensor(labels)
            # No feature is considered for baselines
            x =  torch.ones((point_cloud.shape[0], 3), dtype=torch.float)
            data = Data(pos=point_cloud, x=x, y=labels)
            # normalize coords
            data = self.normalization(data)
            if self.configs['augment'] and self.mode == "train":
                # Augment PC if required
                data = self.augmentations(data)
            return data, None  # forced by collate

    def collate_fn(self, batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    dataset = NeonTreeDataset('rgb', path_augment_dict=True, mode='train')
    for i, data in enumerate(dataset):
        import ipdb; ipdb.set_trace()
    # dataset.plot(0)
