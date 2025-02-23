# Handling of satellite data taken from https://github.com/ermongroup/tile2vec

import sys
import yaml
import argparse
import paths

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--c', dest='config_file', default='config.yml')
args = parser.parse_args()

# read in config file
config = yaml.safe_load(open(args.config_file))
task = config['model']['task']
paths = paths.get_paths(task)

sys.path.append('../')
sys.path.append(paths['home_dir'])

import os
import glob
import pandas
import torch
from torch import optim
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from time import time
import random
import numpy as np
from tensorboardX import SummaryWriter
import pickle
import csv
import pandas as pd
from datetime import date
import shutil
import utils
from src.data_utils import *
from src.train import train_model, validate_model
from src.datasets import satellite_dataloader

# read in config file
config = yaml.safe_load(open(args.config_file))
training = config['training']
data = config['data']
eval = config['eval']
pre = config['preprocessing']

# environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cuda = torch.cuda.is_available()

#### Set random seed ####
utils.set_seed(random.randint(0, 10000))

# create necessary directories if they do not already exist
if not os.path.exists(paths['log_dir']):
	os.makedirs(paths['log_dir'])
if not os.path.exists(paths['model_dir']):
	os.makedirs(paths['model_dir'])
save_dir = paths['model_dir'] + config['model']['name']
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

# data parameters
img_type = data['img_type']
img_size = data['img_size']
bands = data['bands']
img_ext = data['img_ext']
augment = training['augment']
augment_type = training['augment_type']
batch_size = training['batch_size']
shuffle = training['shuffle']
num_workers = training['num_workers']
labels_path = data['labels']
satcutmix_alpha = training['sat-cutmix_alpha']
satcutmix_num_pairs = training['sat-cutmix_num_pairs']


##### Preprocessing #####
# Convert tifs to npy - do this once before training (if needed)
if pre['tif2npy']:
	print('\nConverting tif to npy')
	tif2npy(paths['tif_dir'], paths['npy_dir'], img_ext, bands)

# Calculate dataset means/std devs for preprocessing - do this once before training
if pre['calc_channel_means'] or pre['calc_channel_means_stdDevs']:
	split = pre['split']  # which split to calculate means over

	if pre['calc_channel_means']:
		print(f'\n\nCalculating channel means for {split}')
	else:
		print(f'\n\nCalculating channel means & standard deviations for {split}')

	dataloader = satellite_dataloader(img_type, paths['img_dir'], labels_path, split=split,
									  size=img_size, img_ext=img_ext, bands=bands, task=config['model']['task'],
									  augment=False, augment_type=[], batch_size=1,
									  shuffle=False, num_workers=num_workers, means=None)  # means MUST be None

	means, stds = utils.get_channel_mean_stdDev(dataloader, bands)

	print(f'Means: {means}')
	if pre['calc_channel_means_stdDevs']:
		print(f'Standard deviations: {stds}')

	# save means
	img_path = os.path.normpath(paths['img_dir']).split(os.path.sep)[-3]
	today = date.today()
	d = today.strftime('%b-%d-%Y')
	if pre['calc_channel_means']:
		np.savetxt('channel_means_' + img_path + '_' + d + '.txt', means)
	else:
		means_stds = torch.cat((means, stds))
		np.savetxt('channel_means_stds_' + img_path + '_' + d + '.txt', means_stds)

##### Define model #####
if task == 'eurosat':
	from src.resnet_eurosat import make_cnn
	CNN = make_cnn(num_classes=data['num_classes'], in_channels=bands)
else:
	from src.resnet import make_cnn
	CNN = make_cnn(num_classes=data['num_classes'])  # assumed 3 input channels

# define criterion
if config['model']['mode'] == 'regression':
	criterion = nn.MSELoss()
else:
	print(f"Num classes: {data['num_classes']}")
	if data['num_classes'] == 1:
		criterion = nn.BCEWithLogitsLoss(reduction='mean')
	else:
		criterion = nn.CrossEntropyLoss()

if cuda:
	CNN.cuda()
print('\nCuda available: ' + str(cuda))


#### Dataloaders #####
if training['train']:
	train_split = 'train'
	test_split = 'test'
	val_split = 'val'

	# read in means
	train_means = np.loadtxt(paths['means'])
	train_means = tuple(train_means)

	train_dataloader = satellite_dataloader(img_type, paths['img_dir'], labels_path, split=train_split,
											size=img_size, img_ext=img_ext, bands=bands, task=config['model']['task'],
											augment=augment, augment_type=augment_type,
											batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
											means=train_means)
	print('\nTrain Dataloader set up complete.')
	print(len(train_dataloader))

	if training['val']:
		val_dataloader = satellite_dataloader(img_type, paths['img_dir'], labels_path, split=val_split,
											  size=img_size, img_ext=img_ext, bands=bands, task=config['model']['task'],
											  augment=augment, augment_type=augment_type,
											  batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
											  means=train_means)
		print('Val Dataloader set up complete.')

	test_dataloader = satellite_dataloader(img_type, paths['img_dir'], labels_path, split=test_split,
										   size=img_size, img_ext=img_ext, bands=bands, task=config['model']['task'],
										   augment=augment, augment_type=augment_type,
										   batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
										   means=train_means)
	print('Test Dataloader set up complete.')


##### Training #####
if training['train']:
	# print summary
	print('\nModel summary:')
	print(CNN)
	summary(CNN, (bands, img_size, img_size))

	# load saved model
	if config['model']['model_fn']:
		CNN.load_state_dict(torch.load(config['model']['model_fn']))
		print('\nLoaded saved model')

	print(f"Name: {config['model']['name']}")

	if training['save_models']:
		print('\nSaving checkpoints')
	else:
		print('\nNot saving checkpoints')

	lr = float(training['lr'])
	weight_decay = training['weight_decay']
	beta1 = training['beta1']
	lr_gamma = training['lr_gamma']
	optimizer = optim.AdamW(CNN.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)

	if lr_gamma:
		print('Using learning rate scheduler')
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
	else:
		scheduler = None

	print_every = 2000

	# logging
	writer = SummaryWriter(paths['log_dir'] + config['model']['name'])

	# copy config file to model_dir
	shutil.copyfile(args.config_file, os.path.join(save_dir, os.path.basename(args.config_file)))

	train_loss = []
	test_loss = []
	val_loss = []

	if augment_type[0] in ['CutMix', 'Sat-CutMix', 'Sat-SlideMix']:
		mixing_method = augment_type[0]
	else:
		mixing_method = None
	print('\nMixing Method:')
	print(mixing_method)

	t0 = time()
	for epoch in range(training['epochs_start'], training['epochs_end']):
		if training['train']:

			avg_loss_train = train_model(CNN, cuda, train_dataloader, optimizer,
										 epoch, criterion, data['num_classes'], config['model']['mode'],
										 print_every, mixing_method, satcutmix_alpha, satcutmix_num_pairs)
			train_loss.append(avg_loss_train)
			writer.add_scalar('loss/train', avg_loss_train, epoch)

		if training['test']:
			avg_loss_test = validate_model(CNN, cuda, val_dataloader, epoch,
										   criterion, data['num_classes'])
			test_loss.append(avg_loss_test)
			writer.add_scalar('loss/test', avg_loss_test, epoch)

		if training['val']:
			avg_loss_val = validate_model(CNN, cuda, val_dataloader, epoch,
										  criterion, data['num_classes'])
			val_loss.append(avg_loss_val)
			writer.add_scalar('loss/val', avg_loss_val, epoch)

		if epoch % 50 == 0 and scheduler:
			print("STEPPING")
			scheduler.step()

		if training['save_models'] & (epoch % 5 == 0):
			print('Saving')
			save_name = 'CNN' + str(epoch) + '.ckpt'
			model_path = os.path.join(save_dir, save_name)
			torch.save(CNN.state_dict(), model_path)
			if training['train']:
				with open(save_dir + '/train_loss.p', 'wb') as f:
					pickle.dump(train_loss, f)
			if training['test']:
				with open(save_dir + '/test_loss.p', 'wb') as f:
					pickle.dump(test_loss, f)

		##### Evaluation #####
		results = []

		# Calculate model R^2 (coefficient of determination)
		if eval['r2'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating R^2 for {split}')
			r2 = utils.calc_r2(CNN, test_dataloader, cuda)
			results.append(f'R2 - epoch{str(epoch)}: {str(r2)}')

		# Calculate mean squared error
		if eval['mse'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating MSE for {split}')
			mean_sq_loss = utils.calc_mse(CNN, test_dataloader, cuda)
			results.append(f'MSE - epoch{str(epoch)}: {str(mean_sq_loss)}')

		# Calculate mean absolute error
		if eval['mae'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating MAE for {split}')
			mean_abs_loss = utils.calc_mae(CNN, test_dataloader, cuda)
			results.append(f'MAE - epoch{str(epoch)}: {str(mean_abs_loss)}')

		# Calculate accuracy
		if eval['acc'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating accuracy for {split}')
			acc = utils.calc_acc(CNN, test_dataloader, cuda, data['num_classes'])
			results.append(f'Accuracy - epoch{str(epoch)}: {str(acc)}')

		# Calculate precision, recall & F1-score
		if eval['pr'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating precision & recall for {split}')

			macro, weighted = utils.calc_PR(CNN, test_dataloader, cuda, data['num_classes'])
			results.append(f'Macro - epoch{str(epoch)}: {str(macro)}')
			results.append(f'Weighted - epoch{str(epoch)}: {str(weighted)}')

		# print time
		if epoch % 5 == 0:
			print(f'Training time: {(time() - t0):.2f} sec')

		# save results to txt
		txt = os.path.join(save_dir, 'results_' + config['model']['name'] + '.txt')
		with open(txt, 'a') as f:
			for line in results:
				f.write(f'{line}\n')

	if eval['r2']:
		print(f'R2 = {str(r2)}')
	if eval['mse']:
		print(f'MSE = {str(mean_sq_loss)}')
	if eval['mae']:
		print(f'MAE = {str(mean_abs_loss)}')
	if eval['acc']:
		print(f'Accuracy = {str(acc)}')
	if eval['pr']:
		print(f'Macro = {str(macro)}')
		print(f'Weighted = {str(weighted)}')
	print('Done training!')


