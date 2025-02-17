from torchvision import models
import torch.nn as nn


def make_cnn(num_classes):
	"""
	Returns a ResNet18 model with the specified number of
	input channels and feature dimension.
	"""

	class Identity(nn.Module):
		def __init__(self):
			super(Identity, self).__init__()

		def forward(self, x):
			return x

	resnet = models.resnet18(weights='DEFAULT')

	#(fc): Linear(in_features=512, out_features=1000, bias=True)
	resnet.fc = nn.Sequential(
		nn.Linear(in_features=512, out_features=64, bias=True),
		nn.ReLU(inplace=True),
		nn.Linear(in_features=64, out_features=num_classes, bias=True)
	)

	return resnet
