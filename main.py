#!/usr/bin/env python3

import random
import argparse
import numpy
import torch
import pytorch_lightning as pl
import torchvision
import torchmetrics


def balance_classes(subset: torch.utils.data.Subset, balance: list):
	class_indices = [
		[index for index in subset.indices if subset.dataset.targets[index] == c]
		for c, _ in enumerate(subset.dataset.classes)
	]
	ref = min(len(indices) / balance[c] for c, indices in enumerate(class_indices))
	balanced_indices = [random.sample(indices, int(ref * balance[c])) for c, indices in enumerate(class_indices)]
	subset.indices = sum(balanced_indices, [])


def label_indices(datamodule: pl.LightningDataModule, indices: list):
	datamodule.data_train.indices += indices
	datamodule.data_unlabeled.indices = [index for index in datamodule.data_unlabeled.indices if index not in indices]


def label_randomly(datamodule: pl.LightningDataModule, amount: int):
	chosen_indices = random.sample(datamodule.data_unlabeled.indices, amount)
	label_indices(datamodule, chosen_indices)


def label_uncertain(datamodule: pl.LightningDataModule, amount: int, model: pl.LightningModule):
	uncertainty_list = []
	for batch in datamodule.unlabeled_dataloader():
		x, _ = batch
		y_hat, _ = model(x)
		preds = torch.nn.functional.softmax(y_hat, 1)
		uncertainty_list.append(-(preds * preds.log()).sum(1))

	uncertainty = torch.cat(uncertainty_list)
	top_uncertainties, top_indices = uncertainty.topk(amount)
	chosen_indices = [datamodule.data_unlabeled.indices[i] for i in top_indices]
	label_indices(datamodule, chosen_indices)


def label_highest_loss(datamodule: pl.LightningDataModule, amount: int, model: pl.LightningModule):
	uncertainty_list = []
	for batch in datamodule.unlabeled_dataloader():
		x, _ = batch
		_, losses_hat = model(x)
		uncertainty_list.append(losses_hat)

	uncertainty = torch.cat(uncertainty_list)
	top_uncertainties, top_indices = uncertainty.topk(amount)
	chosen_indices = [datamodule.data_unlabeled.indices[i] for i in top_indices]
	label_indices(datamodule, chosen_indices)


def label_core_set(datamodule: pl.LightningDataModule, amount: int, model: pl.LightningModule):
	# Each time, get the unlabeled data point with the largest minimum distance to a labeled data point

	batch_size = datamodule.hparams.eval_batch_size

	for i in range(amount):
		max_min_dist = 0
		cache_labeled = []

		for batch_labeled in datamodule.train_dataloader():
			x_labeled, _ = batch_labeled
			cache_labeled.append(model.convolutional(x_labeled)**2)
	
		for batch_num, batch_unlabeled in enumerate(datamodule.unlabeled_dataloader()):
			x_unlabeled, _ = batch_unlabeled
			features_unlabeled = model.convolutional(x_unlabeled)

			print(f"Label {i}, point {batch_num*batch_size:05d}", end="\r")

			# TODO With some magic this could be made 1 loop which is probably a lot faster
			for item_num, squares_unlabeled in enumerate(features_unlabeled**2):
				for squares_labeled in cache_labeled:
					# Square root was omitted for efficiency
					cur_min_dist = (squares_labeled - squares_unlabeled).min()
				
				if cur_min_dist > max_min_dist:
					max_min_dist = cur_min_dist
					max_id = batch_size * batch_num + item_num
	
	label_indices(datamodule, [max_id])



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
			balance_classes(self.data_unlabeled, self.hparams.class_balance)
			self.data_train = torch.utils.data.Subset(data_full, [])
			label_randomly(self, self.hparams.initial_labels)

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
			num_workers=4
		)

	def val_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_val,
			batch_size=self.hparams.eval_batch_size,
			num_workers=4
		)

	def test_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_test,
			batch_size=self.hparams.eval_batch_size,
			num_workers=4
		)

	def unlabeled_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_unlabeled,
			batch_size=self.hparams.eval_batch_size,
			num_workers=4
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



def main():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	# Model related
	parser.add_argument(
		'--learning-rate', type=float, default=1e-4,
		help="Multiplier used to tweak model parameters"
	)
	parser.add_argument(
		'--train-batch-size', type=int, default=16,
		help="Batch size used for training the model"
	)
	parser.add_argument(
		'--min-epochs', type=int, default=25,
		help="Minimum epochs to train before switching to the early stopper"
	)

	# Active learning related
	parser.add_argument(
		'--early-stopping-patience', type=int, default=5,
		help="Epochs to wait before stopping training and asking for new data"
	)
	parser.add_argument( # This should probably be made dataset-independant
		'--class-balance', type=list, default=[0.1]*5 + [1.0]*5,
		help="List of class balance multipliers"
	)
	parser.add_argument(
		'--aquisition-method', type=str, default='random',
		choices=['random', 'uncertain', 'learning-loss', 'core-set'],
		help="The unlabeled data aquisition method to use"
	)
	parser.add_argument(
		'--initial-labels', type=int, default=500,
		help="The amount of initially labeled datapoints"
	)
	parser.add_argument(
		'--batch-budget', type=int, default=100,
		help="The amount of datapoints to be labeled per aquisition step"
	)
	parser.add_argument( # TODO Make this dependant on aquisition method
		'--learning-loss-factor', type=float, default=0.1,
		help="Multiplier used on top of the learning rate for the additional learning loss"
	)

	# Device related
	parser.add_argument(
		'--data-dir', type=str, default='./datasets',
		help="Multiplier used to tweak model parameters"
	)
	parser.add_argument(
		'--eval-batch-size', type=int, default=256,
		help="Batch size used for evaluating the model"
	)

	args = parser.parse_args()

	early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
		monitor="validation classification loss",
		patience=args.early_stopping_patience
	)
	trainer = pl.Trainer(
		log_every_n_steps=10,
		min_epochs=args.min_epochs,
		max_epochs=-1,
		callbacks=[early_stopping_callback]
	)
	model = ALModel28(**vars(args))
	mnist = MNISTDataModule(**vars(args))

	# TODO Think of a more appropriate limit
	for _ in range(20):
		trainer.fit(model, mnist)
		trainer.test(model, mnist)

		# TODO Could this be moved to on_train_end?
		early_stopping_callback.best_score = torch.tensor(numpy.Inf)

		# TODO Would it be possible to do this in a callback?
		with torch.no_grad():
			if args.aquisition_method == 'random':
				label_randomly(mnist, args.batch_budget)
			elif args.aquisition_method == 'uncertain':
				label_uncertain(mnist, args.batch_budget, model)
			elif args.aquisition_method == 'learning-loss':
				label_highest_loss(mnist, args.batch_budget, model)
			elif args.aquisition_method == 'core-set':
				label_core_set(mnist, args.batch_budget, model)
			else:
				raise ValueError('Given aquisition method is not available')


if __name__ == "__main__":
	main()
