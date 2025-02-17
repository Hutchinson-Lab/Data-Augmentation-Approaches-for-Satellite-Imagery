# Handling of satellite data taken from https://github.com/ermongroup/tile2vec

import os
from time import time
from src.datasets import satellite_dataloader
import utils
import numpy as np
import rasterio


def clip_and_scale_image(img, img_type='naip'):
    if img_type in ['naip', 'rgb']:
        return img / 255

    elif img_type =='eurosat':  # eurosat images scaled from 0-2750 to 1-255
        return (np.clip(img, 0, 2750) / 2750 * 255).astype(np.uint8)   # need range to be 0-255 for data augmentation


def load_tif_npy(img_fn, bands, bands_only=False):
    img = np.load(img_fn)
    if bands_only and bands > 1:  # if bands == 1, shape will be [:,:]
            img = img[:,:,:bands]
    return img


def load_rgb(img_fn, bands, bands_only=False, is_npy=True):
    if is_npy:
        return load_tif_npy(img_fn, bands, bands_only)
    obj = gdal.Open(img_fn)
    img = obj.ReadAsArray().astype(np.uint8)
    del obj  # close GDAL dataset

    if bands_only and bands > 1:  # if bands == 1, shape will be [:,:]
        img = img[0:bands]  # only select the first X bands
        img = np.moveaxis(img, 0, -1)

    return img


def load_tif(img_fn, bands, img_type, bands_only=False, is_npy=True):
    if is_npy:
        return load_tif_npy(img_fn, bands, bands_only)

    obj = rasterio.open(img_fn)
    img = obj.read()  # ndarray (bands, height, width)
    obj.close()

    if bands_only and bands > 1:  # if bands == 1, shape will be [:,:]
        img = img[0:bands]  # only select the first X bands
        img = np.moveaxis(img, 0, -1)

    img = clip_and_scale_image(img, img_type)
    return img


def tif2npy(tif_dir, npy_dir, img_type, bands):
    count = 0

    for img in os.listdir(tif_dir):
        img_name = img.split('.tif')[0]
        print(img_name)
        if img_type == 'rgb':
            img = load_rgb(tif_dir + img, bands, bands_only=True, is_npy=False)
        else:
            img = load_tif(tif_dir + img, bands, img_type, bands_only=True, is_npy=False)
        if (img.shape[0] >= 48 and img.shape[1] >= 48):
            np.save(os.path.join(npy_dir, img_name + '.npy'), img)

        count += 1
    print(f'Saved  {str(count)} images to {npy_dir}')


def calc_channel_means(img_type, split, paths):
    print(f'\n\nCalculating channel means for {split}')

    dataloader = satellite_dataloader(img_type, paths['img_dir'], paths['labels'], split=split, size=img_size,
                                      bands=bands, augment=False, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers, means=None)  # means MUST be None

    means, _ = get_channel_mean_stdDev(dataloader)

    print(f'Means: {means}')

    # save means
    img_path = os.path.normpath(paths.png_dir).split(os.path.sep)[-1]
    today = date.today()
    d = today.strftime('%b-%d-%Y')
    np.savetxt('channel_means' + img_path + '_' + d + '.txt', means)


def calc_channel_means_stdDevs(img_type, split, paths):
    print(f'\n\nCalculating channel means & standard deviations for {split}')

    dataloader = satellite_dataloader(img_type, paths['png_dir'], paths['labels'], split=split, size=img_size,
                                      bands=bands, augment=False, batch_size=1,
                                      shuffle=False, num_workers=num_workers, means=None)  # means MUST be None

    means, stdDevs = get_channel_mean_stdDev(dataloader)

    print(f'Means: {means}')
    print(f'Standard Deviations: {stdDevs}')

    # save means
    img_path = os.path.normpath(paths.png_dir).split(os.path.sep)[-1]
    today = date.today()
    d = today.strftime('%b-%d-%Y')
    np.savetxt('channel_means' + img_path + '_' + d + '.txt', means + stdDevs)
