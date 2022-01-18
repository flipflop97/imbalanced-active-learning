
import torch
import torchvision

import data_utils
import modules_general


class CIFAR10DataModule(modules_general.UALDataModule):
	def __init__(self, **kwargs):
		super().__init__()


	def prepare_data(self):
		torchvision.datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
		torchvision.datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)


	def setup(self, stage:str=None):
		if stage == "fit" or stage == "validate" or stage is None:
			data_full = torchvision.datasets.CIFAR10(
				self.hparams.data_dir,
				train=True,
				transform=self.transform
			)

			self.data_unlabeled, self.data_val = torch.utils.data.random_split(
				data_full,
				[40000, 10000]
			)
			# TODO This could be baked into the general module too
			data_utils.balance_classes(self.data_unlabeled, self.hparams.class_balance)
			self.data_train = torch.utils.data.Subset(data_full, [])
			data_utils.label_randomly(self, self.hparams.initial_labels)

		if stage == "test" or stage is None:
			self.data_test = torchvision.datasets.CIFAR10(
				self.hparams.data_dir,
				train=False,
				transform=self.transform
			)



class CIFAR10Model(modules_general.UALModel):
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
