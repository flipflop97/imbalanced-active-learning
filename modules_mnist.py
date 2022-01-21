import torch
import torchvision

import modules_general


class MNISTDataModule(modules_general.UALDataModule):
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



class MNISTModel(modules_general.UALModel):
	def __init__(self, **kwargs):
		super().__init__()

		self.convolutional = torch.nn.Sequential(
			torch.nn.Conv2d(1, 6, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
			torch.nn.Conv2d(6, 16, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
			torch.nn.Flatten(1),
			torch.nn.Linear(16*5*5, 128), torch.nn.ReLU()
		)

		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(128, 64), torch.nn.ReLU(),
			torch.nn.Linear(64, 10)
		)

		self.loss = torch.nn.functional.cross_entropy
