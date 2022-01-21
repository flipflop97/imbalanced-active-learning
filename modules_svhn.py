import torch
import torchvision

import modules_general


class SVHNDataModule(modules_general.IALDataModule):
	def __init__(self, **kwargs):
		super().__init__()


	def prepare_data(self):
		torchvision.datasets.SVHN(self.hparams.data_dir, split='train', download=True)
		torchvision.datasets.SVHN(self.hparams.data_dir, split='test', download=True)


	def get_data_train(self):
		data = torchvision.datasets.SVHN(
			self.hparams.data_dir,
			split='train',
			transform=self.transform
		)

		# Needed for intra-dataset consistency
		data.classes = list(map(str, range(10)))
		data.targets = data.labels

		return data


	def get_data_test(self):
		data = torchvision.datasets.SVHN(
			self.hparams.data_dir,
			split='test',
			transform=self.transform
		)

		# Needed for intra-dataset consistency
		data.classes = list(map(str, range(10)))
		data.targets = data.labels

		return data



class SVHNModel(modules_general.IALModel):
	def __init__(self, **kwargs):
		super().__init__()

		self.convolutional = torch.nn.Sequential(
			torch.nn.Conv2d(3, 6, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
			torch.nn.Conv2d(6, 16, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
			torch.nn.Flatten(1),
			torch.nn.Linear(16*6*6, 128), torch.nn.ReLU()
		)

		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(128, 64), torch.nn.ReLU(),
			torch.nn.Linear(64, 10)
		)

		self.loss = torch.nn.functional.cross_entropy
