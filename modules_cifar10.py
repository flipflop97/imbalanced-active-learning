import torchvision

import modules_general


class CIFAR10DataModule(modules_general.IALDataModule):
	def __init__(self, **kwargs):
		super().__init__()


	def prepare_data(self):
		torchvision.datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
		torchvision.datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)


	def get_data_train(self):
		return torchvision.datasets.CIFAR10(
			self.hparams.data_dir,
			train=True,
			transform=self.transform
		)


	def get_data_test(self):
		return torchvision.datasets.CIFAR10(
			self.hparams.data_dir,
			train=False,
			transform=self.transform
		)