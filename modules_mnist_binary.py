
import torch
import torchvision

import modules_general


class MNISTBinaryDataModule(modules_general.UALDataModule):
	def __init__(self, **kwargs):
		super().__init__()


	def prepare_data(self):
		torchvision.datasets.MNIST(self.hparams.data_dir, train=True, download=True)
		torchvision.datasets.MNIST(self.hparams.data_dir, train=False, download=True)


	def get_data_train(self):
		data = torchvision.datasets.MNIST(
			self.hparams.data_dir,
			train=True,
			transform=self.transform
		)

		# Change labels to even (0) or odd (1) numbers
		data.targets %= 2
		data.classes = ['even', 'odd']

		return data


	def get_data_test(self):
		data = torchvision.datasets.MNIST(
			self.hparams.data_dir,
			train=False,
			transform=self.transform
		)

		# Change labels to even (0) or odd (1) numbers
		data.targets %= 2
		data.classes = ['even', 'odd']

		return data



class MNISTBinaryModel(modules_general.UALModel):
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
			torch.nn.Linear(64, 1)
		)

		self.loss = lambda pred, target, *args, **kwargs: \
			torch.nn.functional.binary_cross_entropy_with_logits(pred, target.float(), *args, **kwargs)
