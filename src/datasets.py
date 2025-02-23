# Handling of satellite data taken from https://github.com/ermongroup/tile2vec

from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import random
import pandas as pd
import imageio
from src.augmentation import sat_trivial, automated, color_geo_transformations, moco_aug_plus


class SatelliteDataset(Dataset):

    def __init__(self, img_dir, label_fn, split, size, img_type, img_ext, bands, task, img_size,
                 augment, augment_type=[], means=None):
        self.img_dir = img_dir
        self.size = size
        self.img_type = img_type
        self.img_files = glob.glob(os.path.join(self.img_dir, '*'))
        self.img_ext = img_ext
        self.bands = bands
        self.task = task
        self.img_size = img_size
        self.augment = augment
        self.augment_type = augment_type
        self.means = means
        self.label_fn = label_fn
        self.ids, self.labels = get_ids_and_labels_from_npy(split, label_fn)


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        single_id = self.ids[idx]
        single_label = self.labels[idx]

        try:
            if self.img_ext == 'png':
                img = imageio.imread(os.path.join(self.img_dir, single_id + '.png'))
            elif self.img_ext == 'jpg':
                img = imageio.imread(os.path.join(self.img_dir, single_id + '.jpg'))
            elif self.img_ext == 'npy':
                img = np.load(os.path.join(self.img_dir, single_id + '.npy'))

            if self.bands == 1:  # (X, Y) --> (1, X, Y)
                img = np.expand_dims(img, axis=0)
            else:  # reshape: (X, Y, channel) --> (channel, X, Y)
                img = np.moveaxis(img, -1, 0)
            img = img[:, 0:self.size, 0:self.size]

            # define which type of augmentation to perform
            automated_methods = ['Auto_ImageNet', 'Auto_CIFAR', 'Auto_SVHN', 'Random', 'Trivial']
            mixing_methods = ['CutMix', 'Sat-CutMix', 'Sat-SlideMix']  # augmentation is performed in train.py

            if self.augment_type[0] == 'Sat-Trivial':
                transform = sat_trivial(self.img_size, self.bands, self.means)
            elif self.augment_type[0] == 'MoCo_aug_plus':
                transform = moco_aug_plus(self.img_size, self.bands, self.means)
            elif self.augment_type[0] in automated_methods:
                transform = automated(self.augment_type, self.task, self.bands, self.means)
            elif self.augment_type[0] in mixing_methods:
                transform = color_geo_transformations(['identity', 'flip', 'rotate'], self.img_type,
                                                      self.bands, self.means)
            else:
                transform = color_geo_transformations(self.augment_type, self.img_type,
                                                      self.bands, self.means)
            img = transform(img)

            return img, single_label, single_id

        except Exception as e:
            raise Exception(f'Could not open {single_id}')
            print(e)
            


def satellite_dataloader(img_type, img_dir, label_fn, split, size, img_ext, bands, task,
                         augment=True, augment_type=[], batch_size=4,
                         shuffle=True, num_workers=4, means=None):
    """
    Returns a DataLoader with either RGB multispectral images.

    """
    assert img_type in ['rgb', 'sentinel-2']

    dataset = SatelliteDataset(img_dir, label_fn, split, size, img_type, img_ext, bands, task, size,
                               augment, augment_type, means)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True)
    return dataloader


def get_ids_and_labels_from_npy(split, label_fn):
    if 'elevation' in label_fn or 'treecover' in label_fn or 'nightlights' in label_fn or 'population' in label_fn:
        label_col = 'label_normalized'
    else:
        label_col = 'label'

    label_df = pd.read_csv(label_fn)

    if split == 'all':
        ids = label_df['id'].tolist()  # convert column to list
        labels = label_df[label_col].tolist()
    else:
        ids = label_df.loc[label_df['fold'] == split, ['id']]
        ids = ids['id'].tolist()
        labels = label_df.loc[label_df['fold'] == split, label_col]
        labels = labels.tolist()

    print(f'split: {split}')
    print(f'len ids: {str(len(ids))}')
    return ids, labels
