from tqdm import tqdm
import numpy as np
import laspy
import rasterio
import pickle
from rasterio import features
import geopandas as gpd
import xmltodict
import yaml
from pathlib import Path
from PIL import Image
import einops

from albumentations.augmentations.crops.functional import crop_bbox_by_coords
from shapely.geometry import Polygon, box

from utils.paths import Paths

class Tilerizer:

    def __init__(self, data_path, annot_path=None, tiles_path=None, masks_path=None, task='detection', modality='image'):
        self.data_path = data_path
        self.data_extension = self.data_path.suffix
        self.annot_path = annot_path
        self.task = task
        self.modality = modality
        self.metadata, self.data = self._load_data()
        if self.annot_path:
            self.labels, self.categories, self.agb = self._load_annots()
        if self.data is None:
            print('Data is None, problem occurred during loading')
        else:
            self._create_folders()

    def _load_data(self):
        if self.modality == 'image':
            if self.data_extension == '.png':
                with Image.open(self.data_path) as img:
                    data = np.array(img)
                    data = einops.rearrange(data, 'h w c -> c h w')
                metadata = None
            elif self.data_extension == '.tif':
                with rasterio.open(self.data_path) as src:
                    metadata = src.profile
                    data = src.read()
            else:
                raise Exception('Data format {} not supported yet.'.format(self.data_extension))
        elif self.modality == 'point_cloud':
            try:
                with laspy.open(self.data_path) as pc_file:
                    data = pc_file.read()
                metadata = None
            except Exception:
                return None, None
        else:
            raise Exception('Modality {} is not supported yet.'.format(self.modality))
        return metadata, data

    def _load_annots(self):
        if self.task == 'detection':
            if self.annot_path.suffix.lower() == '.xml':
                with open(self.annot_path, 'r') as annotation_file:
                    annotation = xmltodict.parse(annotation_file.read())
                labels = []
                if isinstance(annotation['annotation']['object'], list):
                    for bbox in annotation['annotation']['object']:
                        xmin = bbox['bndbox']['xmin']
                        ymin = bbox['bndbox']['ymin']
                        xmax = bbox['bndbox']['xmax']
                        ymax = bbox['bndbox']['ymax']
                        labels.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                else:
                    xmin = annotation['annotation']['object']['bndbox']['xmin']
                    ymin = annotation['annotation']['object']['bndbox']['ymin']
                    xmax = annotation['annotation']['object']['bndbox']['xmax']
                    ymax = annotation['annotation']['object']['bndbox']['ymax']
                    labels.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                categories = None
                agb = None
            elif self.annot_path.suffix == '.csv':
                import pandas as pd
                annots = pd.read_csv(self.annot_path)
                annots = annots[annots['img_path'] == self.data_path.name]
                labels = annots[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
                if 'group' in annots.columns:
                    categories = annots['group'].to_numpy()
                else:
                    categories = None
                if 'AGB' in annots.columns:
                    agb = annots['AGB'].to_numpy()
                else:
                    categories = None
            else:
                raise Exception('Annotation format {} not supported yet.'.format(self.annot_path.suffix))
        else:
            raise Exception('Task {} not implemented yet.'.format(self.task))
        return labels, categories, agb

    def _create_folders(self):
        if self.modality == 'image':
            self.tiles_folder = self.data_path.parent / 'tiles' / self.data_path.stem
            self.tiles_folder.mkdir(parents=True, exist_ok=True)
            # self.masks_folder = self.tif_path.parent / 'masks' / self.tif_path.stem
            # self.masks_folder.mkdir(parents=True, exist_ok=True)
            self.labels_folder = self.data_path.parent / 'labels' / self.data_path.stem
            self.labels_folder.mkdir(parents=True, exist_ok=True)
        elif self.modality == 'point_cloud':
            self.pcs_folder = self.data_path.parent / 'sub_point_clouds' / self.data_path.stem
            self.pcs_folder.mkdir(parents=True, exist_ok=True)
        else:
            raise Exception('Modality {} is not supported yet.'.format(self.modality))

    def _crop_labels(self, crop_coords):
        if self.task == 'detection':
            x_min, y_min, x_max, y_max = crop_coords
            bbox_polys = np.array([box(*bbox) for bbox in self.labels])
            crop_poly = box(*crop_coords)
            intersect_bboxes = np.array([bbox.intersection(crop_poly).area/bbox.area for bbox in bbox_polys])
            idx_full_bboxes = np.where(intersect_bboxes == 1.)[0]
            idx_partial_boxes = np.where((intersect_bboxes > 0.) & (intersect_bboxes < 1.))[0]
            # Crop boxes with partial intersection
            subboxes = np.array([bbox.intersection(crop_poly) for bbox in bbox_polys[idx_partial_boxes]])
            # Check if all boxes are in the crop
            for bbox in subboxes:
                assert bbox.intersection(crop_poly).area/bbox.area == 1., 'A box has not been cropped well!'
            valid_boxes = np.hstack((bbox_polys[idx_full_bboxes], subboxes))
            # From polygones to boxes
            valid_boxes = [list(box_poly.bounds) for box_poly in valid_boxes]
            norm_valid_boxes = [[valid_box[0]-x_min, valid_box[1]-y_min, valid_box[2]-x_min, valid_box[3]-y_min]
                                for valid_box in valid_boxes]
            idx_boxes = np.hstack((idx_full_bboxes, idx_partial_boxes))
            assert len(norm_valid_boxes) == len(idx_boxes), 'List of boxes and list of idx have different num of elements.'
            return norm_valid_boxes, idx_boxes
        else:
            raise Exception('Task {} not implemented yet.'.format(self.task))
        
    def create_tiles(self, tile_size=256):
        num_rows = self.data.shape[1]
        num_cols = self.data.shape[2]
        samples = []
        print('Full area size: ', self.data.shape[1:])
        print('Desired tile size: ', tile_size)

        print('Saving tiles')

        for row in tqdm(range(0, num_rows, tile_size)):
            for col in tqdm(range(0, num_cols, tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                tile = self.data[:, window.row_off:window.row_off+window.height,
                                window.col_off:window.col_off+window.width]
                if self.metadata:
                    tile_profile = self.metadata.copy()
                    tile_profile.update({
                        'height': tile.shape[1],
                        'width': tile.shape[2],
                        'transform': rasterio.windows.transform(window, self.metadata['transform'])
                    })

                tile_name = 'tile_{}_{}{}'.format(row, col, self.data_extension)
                if self.data_extension == '.tif':
                    with rasterio.open(self.tiles_folder / tile_name, 'w', **tile_profile) as dst:
                        dst.write(tile)
                elif self.data_extension == '.png':
                    tile = einops.rearrange(tile, 'c h w -> h w c')
                    tile_im = Image.fromarray(tile)
                    tile_im.save(self.tiles_folder / tile_name)

                crop_coords = [window.col_off, window.row_off, window.col_off+window.width, window.row_off+window.height]
                labels, idx_boxes = self._crop_labels(crop_coords)
                label_file_name = 'labels_{}_{}.pkl'.format(row, col)
                if self.categories is not None:
                    labels = {'boxes': labels,
                              'categories': self.categories[idx_boxes]}
                    if self.agb is not None:
                        labels['AGB'] = self.agb[idx_boxes]
                with open(self.labels_folder / label_file_name, 'wb') as f:
                    pickle.dump(labels, f)

                sample = {'image': self.tiles_folder / tile_name,
                          'labels': self.labels_folder / label_file_name}
                samples.append(sample)

        paths_file_name = self.data_path.stem + '_paths.pkl'
        with open(self.data_path.parent / paths_file_name, 'wb') as f:
            pickle.dump(samples, f)
        return samples

    def create_subpointcloud(self, nb_max_points=100000):
        if self.data is None:
            print('*****')
            print('Cannot create sub point cloud, the following file does not exists: {}'.format(self.data_path))
            print('*****')
            return None
        x_window, y_window = self._define_pc_window(nb_max_points)
        num_rows = int(np.ceil(self.data.x.max() - self.data.x.min()))
        num_cols = int(np.ceil(self.data.y.max() - self.data.y.min()))
        samples = []
        print('Full area size: ({}, {})'.format(num_rows, num_cols))
        print('Desired x window size: ', x_window)
        print('Desired y window size: ', y_window)

        print('Saving sub point clouds')
        for row in tqdm(range(0, num_rows, x_window)):
            for col in tqdm(range(0, num_cols, y_window)):
                window = rasterio.windows.Window(col, row, y_window, x_window)
                x_min = self.data.x.min() + window.row_off
                y_min = self.data.y.min() + window.col_off
                sub_pc = self.data[np.where((self.data.x >= x_min) &
                                            (self.data.x < x_min+window.height) &
                                            (self.data.y >=  y_min) &
                                            (self.data.y < y_min+window.width))[0]]

                sub_pc_name = 'sub_pc_{}_{}.laz'.format(row, col)
                try:
                    sub_pc.write(self.pcs_folder / sub_pc_name)
                    sample = {'sub_pc': self.pcs_folder / sub_pc_name}
                    samples.append(sample)
                except AttributeError:
                    # Empty point cloud
                    print('Try to save empty point cloud, skipped')
                    pass

        paths_file_name = self.data_path.stem + '_paths.pkl'
        with open(self.data_path.parent / paths_file_name, 'wb') as f:
            pickle.dump(samples, f)
        return samples

    def _define_pc_window(self, nb_max_points):
        total_nb_points = len(self.data.x)
        x_min = self.data.x.min()
        x_max = self.data.x.max()
        y_min = self.data.y.min()
        y_max = self.data.y.max()
        x_dist = 1
        y_dist = 1
        if total_nb_points <= nb_max_points:
            x_dist = int(np.ceil(x_max - x_min))
            y_dist = int(np.ceil(y_max - y_min))
            return x_dist, y_dist
        while len(np.where((self.data.x <= x_min+x_dist) & (self.data.y <= y_min+y_dist))[0]) <= nb_max_points:
            x_dist += 1
            y_dist += 1
        # remove the last iteration
        x_dist -= 1
        y_dist -=1
        return x_dist, y_dist


def test_image(neontree_path):
    tif_path = neontree_path / 'training' / 'RGB' / '2018_BART_4_322000_4882000_image_crop.tif'
    annot_path = neontree_path / 'annotations' / '2018_BART_4_322000_4882000_image_crop.xml'
    tilerizer = Tilerizer(tif_path, annot_path)
    samples = tilerizer.create_tiles()

def test_pointcloud(neontree_path):
    with open(neontree_path / 'lidar_annots_paths.yml', 'r') as fp:
        paths = yaml.safe_load(fp)
    pc_path = Path(paths['training']['annot_laz'][0])
    tilerizer = Tilerizer(pc_path, task='segmentation', modality='point_cloud')
    samples = tilerizer.create_subpointcloud()

if __name__ == '__main__':
    neontree_path = Paths().get()['neontree']
    # test_image(neontree_path)
    test_pointcloud(neontree_path)
    import ipdb; ipdb.set_trace()
