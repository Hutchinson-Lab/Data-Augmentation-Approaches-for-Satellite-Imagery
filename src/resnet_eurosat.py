from torchvision import models
import torch.nn as nn
import math


def make_cnn(num_classes, in_channels):
	"""
	Returns a ResNet18 model with the specified number of
	input channels and feature dimension.
	"""

	class Identity(nn.Module):
		def __init__(self):
			super(Identity, self).__init__()

		def forward(self, x):
			return x


	resnet = models.resnet18(pretrained=True)
	resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False)

	# repeat conv1 weights RGBRGB... and scale 3/C per https://arxiv.org/abs/2306.09424
	w = resnet.conv1.state_dict()['weight']  # [64, 3, 7, 7]
	n = math.ceil(in_channels / 3)  # how many times to repeat
	w = w.repeat((1, n, 1, 1))
	w = w[:, 0:in_channels, :, :]  # [64, in_channels, 7, 7]
	w = w * 3 / in_channels

	resnet.conv1.load_state_dict({'weight': w})

	# final module
	#(fc): Linear(in_features=512, out_features=1000, bias=True)
	resnet.fc = nn.Sequential(
		nn.Linear(in_features=512, out_features=64, bias=True),
		nn.ReLU(inplace=True),
		nn.Linear(in_features=64, out_features=num_classes, bias=True)
	)

	return resnet

