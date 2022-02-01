import torch
import torchvision

import modules_general


class MNISTDataModule(modules_general.IALDataModule):
	def __init__(self, **kwargs):
		super().__init__()


	def prepare_data(self):
		torchvision.datasets.MNIST(self.hparams.data_dir, train=True, download=True)
		torchvision.datasets.MNIST(self.hparams.data_dir, train=False, download=True)


	def get_data_train(self):
		return torchvision.datasets.MNIST(
			self.hparams.data_dir,
			train=True,
			transform=self.transform
		)


	def get_data_test(self):
		return torchvision.datasets.MNIST(
			self.hparams.data_dir,
			train=False,
			transform=self.transform
		)