import numpy as np
import os
import glob
import random
from time import time
import src.data_utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from time import sleep
from skimage import io
from skimage.feature import local_binary_pattern
import sklearn.metrics as metrics
import src.datasets
import pandas as pd
import cv2


def get_ids_and_labels_from_npy(split, label_fn):
    if 'elevation' in label_fn or 'treecover' in label_fn or 'nightlights' in label_fn or 'population' in label_fn:
        label_col = 'label_normalized'
    else:
        label_col = 'label'
    col_fn = os.path.splitext(label_fn)[0] + '_columns.npy'
    label_df = np.load(label_fn, allow_pickle=True)
    label_df_cols = np.load(col_fn, allow_pickle=True)
    label_df = pd.DataFrame(label_df, columns=label_df_cols)

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


def get_channel_mean_stdDev (dataloader, bands):
    means, stdDevs = None, None
    names = []
    for img, label, id in dataloader:
        img = img.reshape(img.shape[0], bands, -1)  # flatten img w & h (maintain batch size and channels)
        if means is None:
            means = torch.mean(img, dim=(0, 2))  # reduce over batch size and img pixels (take means over channels)
        else:
            means += torch.mean(img, dim=(0, 2))
        if stdDevs is None:
            stdDevs = torch.std(img, dim=(0, 2))  # reduce over batch size and img pixels (take means over channels)
        else:
            stdDevs += torch.std(img, dim=(0, 2))
        names.append(id)
    means = means / len(dataloader)
    stdDevs = stdDevs / len(dataloader)
    return means, stdDevs


def tif_to_npy():
    count = 0

    for img in os.listdir(paths.tif_dir):
        img_name = img.split('.tif')[0]
        print(img_name)
        if img_type == 'nlcd':
            img = load_nlcd(paths.tif_dir + img)
        elif img_type == 'rgb':
            img = load_rgb(paths.tif_dir + img, bands, bands_only=True, is_npy=False)
        else:
            img = load_tif(paths.tif_dir + img, bands, bands_only=True, is_npy=False)
        if (img.shape[0] >= 48 and img.shape[1] >= 48):
            np.save(os.path.join(paths.npy_dir, img_name + '.npy'), img)

        count += 1
    print('Saved ' + str(count) + ' images to ' + paths.npy_dir)


def csv_to_npy(csv_path, working_dir):
    fn = os.path.splitext(csv_path)[0]
    label_df = pd.read_excel(fn+'.xlsx', header=0)
    np.save(fn + '.npy', label_df)
    print(fn)

    task = os.path.basename(fn)
    fp = os.path.join(working_dir, 'labels', task + '_columns.npy')
    print(fp)
    np.save(fp, list(label_df.columns))

    print(f'Converted {fn} to .npy')


def clip_extent(npy):
    if npy:
        img_names = glob.glob(paths.imgs_to_clip + "*.npy")
    else:
        img_names = glob.glob(paths.imgs_to_clip + "*.tif")

    # Clip and save images
    clip_img(img_names, bands, patch_size=args.extent, patch_per_img=1,  # patch size 200 = 2km x 2km
             centered=True, save=True, verbose=True, npy=npy)

    print("Finished.")


def resize_img(img_path, dim_x, dim_y):
    # get name of all images in img_path
    if dim_x == dim_y:
        outptut_path = img_path + '_' + str(dim_x)
    else:
        outptut_path = img_path + '_' + str(dim_x) + '_' + str(dim_y)
    if not os.path.exists(outptut_path):
        os.makedirs(outptut_path)

    imgs = os.listdir(img_path)
    for img_name in imgs:
        print(img_name)
        img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_UNCHANGED)
        img_resized = cv2.resize(img, dsize=(dim_x, dim_y), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(outptut_path, img_name), img_resized)


# based on https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


### Evalution metrics ###
def get_model_outputs(CNN, dataloader, cuda):
    CNN.eval()
    with torch.no_grad():
        all_predictions = None
        for imgs, labels, ids in dataloader:
            if cuda:
                imgs = imgs.to('cuda')

            if all_predictions is None:
                all_predictions = CNN(imgs)
                all_labels = labels
                all_ids = ids
            else:
                model_outputs = CNN(imgs)
                all_predictions = torch.cat([all_predictions, model_outputs], dim=0)
                all_labels = torch.cat([all_labels, labels], dim=0)
                all_ids += ids

    return all_predictions, all_labels, all_ids


def calc_r2(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    predictions = predictions.detach().cpu().numpy().ravel()
    labels = labels.detach().cpu().numpy()

    r2 = metrics.r2_score(labels, predictions)  # y_true, y_pred
    return r2


def calc_mse(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    predictions = predictions.detach().cpu().numpy().ravel()
    labels = labels.detach().cpu().numpy()
    mse = metrics.mean_squared_error(labels, predictions)
    return mse


def calc_mae(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    predictions = predictions.detach().cpu().numpy().ravel()
    labels = labels.detach().cpu().numpy()
    mae = metrics.mean_absolute_error(labels, predictions)
    return mae


def calc_acc(CNN, dataloader, cuda, num_classes):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    labels = labels.detach().cpu().numpy()

    # convert raw model outputs to classes
    if num_classes == 1:
        m = nn.Sigmoid()  # nn.Softmax()
        sig_preds = m(predictions).detach().cpu().numpy()
        pred_classes = [0 if x < 0.5 else 1 for x in sig_preds]  # 0.5 only because coffee is balanced

    else:
        m = nn.Softmax(dim=1)
        sf_max_preds = m(predictions)
        pred_classes = torch.argmax(sf_max_preds, dim=1)
        pred_classes = pred_classes.detach().cpu().numpy()

    # calc accuracy
    acc = metrics.accuracy_score(labels, pred_classes)
    print(acc)
    return acc


def calc_confusion_matrix(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    labels = labels.detach().cpu().numpy()


    m = nn.Softmax(dim=1)
    sf_max_preds = m(predictions)
    pred_classes = torch.argmax(sf_max_preds, dim=1)
    pred_classes = pred_classes.detach().cpu().numpy()

    # calc accuracy
    confusion = metrics.confusion_matrix(labels, pred_classes)
    return confusion


def calc_PR(CNN, dataloader, cuda, num_classes):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    labels = labels.detach().cpu().numpy()

    # convert raw model outputs to classes
    if num_classes == 1:
        m = nn.Sigmoid()  # nn.Softmax()
        sig_preds = m(predictions).detach().cpu().numpy()
        pred_classes = [0 if x < 0.5 else 1 for x in sig_preds]  # o.5 only becasue coffee is balanced
    else:
        m = nn.Softmax(dim=1)
        sf_max_preds = m(predictions)
        pred_classes = torch.argmax(sf_max_preds, dim=1)
        pred_classes = pred_classes.detach().cpu().numpy()

    # confusion matrix
    rep = metrics.classification_report(labels, pred_classes, digits=3, output_dict=True)
    return rep['macro avg'], rep['weighted avg']
