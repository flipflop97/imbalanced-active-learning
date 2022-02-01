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