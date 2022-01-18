
import torch
import torchmetrics
import torchvision
import pytorch_lightning as pl


class UALDataModule(pl.LightningDataModule):
	def __init__(self, **kwargs):
		super().__init__()

		self.save_hyperparameters()
		self.transform = torchvision.transforms.ToTensor()

		self.data_train = None
		self.data_val = None
		self.data_test = None
		self.data_unlabeled = None


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

	def predict_dataloader(self):
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



class UALModel(pl.LightningModule):
	def __init__(self, **kwargs):
		super().__init__()

		self.save_hyperparameters()

		self.accuracy = torchmetrics.Accuracy()
		self.convolutional = None
		self.classifier = None
		self.loss = None

		if self.hparams.aquisition_method == 'learning-loss':
			self.learning_loss = torch.nn.Sequential(
				torch.nn.Linear(128, 64), torch.nn.ReLU(),
				torch.nn.Linear(64, 1)
			)


	def forward(self, x):
		h = self.convolutional(x)
		preds = self.classifier(h).squeeze(1)

		if self.hparams.aquisition_method == 'learning-loss':
			# TODO should h be detached to avoid double cnn learning or not?
			pred_loss = self.learning_loss(h.detach()).squeeze()
			
			return preds, pred_loss
		else:
			return preds, None


	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat, losses_hat = self(x)

		loss = self.loss(y_hat, y)
		self.log("training classification loss", loss)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = self.loss(y_hat, y, reduction='none')
			loss_loss = torch.nn.functional.mse_loss(losses_hat, losses)
			self.log("training loss loss", loss_loss)

			loss += self.hparams.learning_loss_factor * loss_loss

		return loss

	def on_train_end(self):
		# https://github.com/PyTorchLightning/pytorch-lightning/issues/5007
		self.trainer.fit_loop.current_epoch += 1

		# To force skip early stopping the next epoch
		self.trainer.fit_loop.min_epochs = self.trainer.fit_loop.current_epoch + self.hparams.min_epochs


	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat, losses_hat = self(x)

		loss = self.loss(y_hat, y)
		self.log("validation classification loss", loss)

		accuracy = self.accuracy(y_hat, y)
		self.log("validation classification accuracy", accuracy)

		num_labeled = float(len(self.trainer.datamodule.data_train.indices))
		self.log("labeled data", num_labeled)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = self.loss(y_hat, y, reduction='none')
			loss_loss = torch.nn.functional.mse_loss(losses_hat, losses)
			self.log("validation loss loss", loss_loss)

		return loss


	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat, losses_hat = self(x)

		loss = self.loss(y_hat, y)
		self.log("test classification loss", loss)

		accuracy = self.accuracy(y_hat, y)
		self.log("test classification accuracy", accuracy)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = self.loss(y_hat, y, reduction='none')
			loss_loss = torch.nn.functional.mse_loss(losses_hat, losses)
			self.log("test loss loss", loss_loss)

		return loss


	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
