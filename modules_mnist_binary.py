import torchvision

import modules_general


class MNISTBinaryDataModule(modules_general.IALDataModule):
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
