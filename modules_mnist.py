
import torch
import torchmetrics
import torchvision
import pytorch_lightning as pl

import data_utils


class MNISTDataModule(pl.LightningDataModule):
	def __init__(self, **kwargs):
		super().__init__()

		self.save_hyperparameters()
		self.transform = torchvision.transforms.ToTensor()


	def prepare_data(self):
		torchvision.datasets.MNIST(self.hparams.data_dir, train=True, download=True)
		torchvision.datasets.MNIST(self.hparams.data_dir, train=False, download=True)


	def setup(self, stage:str=None):
		if stage == "fit" or stage == "validate" or stage is None:
			data_full = torchvision.datasets.MNIST(
				self.hparams.data_dir,
				train=True,
				transform=self.transform
			)

			self.data_unlabeled, self.data_val = torch.utils.data.random_split(
				data_full,
				[50000, 10000]
			)
			data_utils.balance_classes(self.data_unlabeled, self.hparams.class_balance)
			self.data_train = torch.utils.data.Subset(data_full, [])
			data_utils.label_randomly(self, self.hparams.initial_labels)

		if stage == "test" or stage is None:
			self.data_test = torchvision.datasets.MNIST(
				self.hparams.data_dir,
				train=False,
				transform=self.transform
			)


	def train_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_train,
			batch_size=self.hparams.train_batch_size,
			shuffle=True,
			num_workers=self.hparams.dataloader_workers
		)

	def val_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_val,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers
		)

	def test_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_test,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers
		)

	def labeled_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_train,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers
		)

	def unlabeled_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_unlabeled,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers
		)



class ALModel28(pl.LightningModule):
	def __init__(self, **kwargs):
		super().__init__()

		self.save_hyperparameters()

		self.accuracy = torchmetrics.Accuracy()

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

		if self.hparams.aquisition_method == 'learning-loss':
			self.learning_loss = torch.nn.Sequential(
				torch.nn.Linear(128, 64), torch.nn.ReLU(),
				torch.nn.Linear(64, 1)
			)


	def forward(self, x):
		h = self.convolutional(x)
		preds = self.classifier(h)

		if self.hparams.aquisition_method == 'learning-loss':
			# TODO should h be detached to avoid double cnn learning or not?
			pred_loss = self.learning_loss(h.detach()).squeeze()
			
			return preds, pred_loss
		else:
			return preds, None


	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat, losses_hat = self(x)

		loss = torch.nn.functional.cross_entropy(y_hat, y)
		self.log("training classification loss", loss)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = torch.nn.functional.cross_entropy(y_hat, y, reduction='none')
			loss_loss = torch.nn.functional.mse_loss(losses_hat, losses)
			self.log("training loss loss", loss_loss)

			loss += self.hparams.learning_loss_factor * loss_loss

		return loss

	def on_train_end(self):
		# https://github.com/PyTorchLightning/pytorch-lightning/issues/5007
		self.trainer.fit_loop.current_epoch += 1

		# To force skip early stopping the next epoch
		self.trainer.fit_loop.min_epochs = self.trainer.fit_loop.current_epoch + 1


	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat, losses_hat = self(x)

		loss = torch.nn.functional.cross_entropy(y_hat, y)
		self.log("validation classification loss", loss)

		accuracy = self.accuracy(y_hat, y)
		self.log("validation classification accuracy", accuracy)

		num_labeled = float(len(self.trainer.datamodule.data_train.indices))
		self.log("labeled data", num_labeled)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = torch.nn.functional.cross_entropy(y_hat, y, reduction='none')
			loss_loss = torch.nn.functional.mse_loss(losses_hat, losses)
			self.log("validation loss loss", loss_loss)

		return loss


	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat, losses_hat = self(x)

		loss = torch.nn.functional.cross_entropy(y_hat, y)
		self.log("test classification loss", loss)

		accuracy = self.accuracy(y_hat, y)
		self.log("test classification accuracy", accuracy)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = torch.nn.functional.cross_entropy(y_hat, y, reduction='none')
			loss_loss = torch.nn.functional.mse_loss(losses_hat, losses)
			self.log("test loss loss", loss_loss)

		return loss


	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
