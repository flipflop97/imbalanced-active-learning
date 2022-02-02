import random
import numpy
import tqdm
import torch
import torchmetrics
import torchvision
import pytorch_lightning as pl

import data_utils


class IALDataModule(pl.LightningDataModule):
	def __init__(self, **kwargs):
		super().__init__()

		self.save_hyperparameters()
		self.transform = torchvision.transforms.ToTensor()

		self.data_train = None
		self.data_val = None
		self.data_test = None
		self.data_unlabeled = None


	def setup(self, stage:str=None):
		if stage == "fit" or stage == "validate" or stage is None:
			data_full = self.get_data_train()

			# Split dataset in unlabeled and validation sets randomly
			size_train = round(len(data_full) * self.hparams.train_split)
			size_val = len(data_full) - size_train
			self.data_unlabeled, self.data_val = torch.utils.data.random_split(
				data_full,
				[size_train, size_val]
			)
			self.data_train = torch.utils.data.Subset(data_full, [])

			data_utils.balance_classes(self.data_unlabeled, self.hparams.class_balance)

			# Label 1 label of each class randomly, then label the rest randomly independant of classes
			self.label_each_class()
			self.label_randomly(self.hparams.initial_labels - len(data_full.classes))

		if stage == "test" or stage is None:
			self.data_test = self.get_data_test()


	def train_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_train,
			batch_size=self.hparams.train_batch_size,
			shuffle=True,
			num_workers=self.hparams.dataloader_workers,
			persistent_workers=True
		)

	def val_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_val,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers,
			persistent_workers=True
		)

	def test_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_test,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers,
			persistent_workers=True
		)

	def predict_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_test,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers,
			persistent_workers=True
		)

	def labeled_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_train,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers,
			persistent_workers=True
		)

	def unlabeled_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_unlabeled,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers,
			persistent_workers=True
		)


	def label_indices(self, indices: list):
		self.data_train.indices += indices
		self.data_unlabeled.indices = [index
			for index in self.data_unlabeled.indices
			if index not in indices
		]


	def label_each_class(self, amount: int = 1):
		return sum([
			random.sample([index
				for index in self.data_unlabeled.indices
				if self.data_unlabeled.dataset.targets[index] == c
			], amount)
			for c, _ in enumerate(self.data_unlabeled.dataset.classes)
		], [])


	def label_randomly(self, amount: int):
		chosen_indices = random.sample(self.data_unlabeled.indices, amount)
		self.label_indices(chosen_indices)


	def label_uncertain(self, amount: int, model: pl.LightningModule):
		uncertainty_list = []
		for batch in tqdm.tqdm(self.unlabeled_dataloader(), desc='Labeling'):
			x, _ = batch
			y_hat, _ = model(x)

			try:
				# Multiclass, softmax
				preds = torch.nn.functional.softmax(y_hat, 1)
				uncertainty_list.append(-(preds*preds.log()).sum(1))
			except IndexError:
				# Binary, sigmoid
				preds = torch.sigmoid(y_hat)
				uncertainty_list.append(-preds*preds.log() - (1-preds)*(1-preds).log())

		uncertainty = torch.cat(uncertainty_list)
		_, top_indices = uncertainty.topk(amount)
		chosen_indices = [self.data_unlabeled.indices[i] for i in top_indices]
		self.label_indices(chosen_indices)


	# Learning Loss for Active Learning
	# Donggeun Yoo, In So Kweon
	def label_highest_loss(self, amount: int, model: pl.LightningModule):
		uncertainty_list = []
		for batch in tqdm.tqdm(self.unlabeled_dataloader(), desc='Labeling'):
			x, _ = batch
			_, losses_hat = model(x)
			uncertainty_list.append(losses_hat)

		uncertainty = torch.cat(uncertainty_list)
		_, top_indices = uncertainty.topk(amount)
		chosen_indices = [self.data_unlabeled.indices[i] for i in top_indices]
		self.label_indices(chosen_indices)


	# Active Learning for Convolutional Neural Networks: A Core-Set Approach
	# Ozan Sener, Silvio Savarese
	def label_core_set(self, amount: int, model: pl.LightningModule):
		# Each time, get the unlabeled data point with the largest minimum distance to a labeled data point

		batch_size = self.hparams.eval_batch_size

		for _ in tqdm.trange(amount, desc='Labeling'):
			max_min_dist = 0
			cache_labeled = []

			for batch_labeled in self.labeled_dataloader():
				x_labeled, _ = batch_labeled
				cache_labeled.append(model.convolutional(x_labeled))

			for batch_index, batch_unlabeled in enumerate(self.unlabeled_dataloader()):
				x_unlabeled, _ = batch_unlabeled
				features_unlabeled = model.convolutional(x_unlabeled)

				min_dist = torch.full([len(features_unlabeled)], numpy.Inf)
				for features_labeled in cache_labeled:
					uu = features_unlabeled.pow(2).sum(1, keepdim=True).T
					ll = features_labeled.pow(2).sum(1, keepdim=True)
					lu = features_labeled @ features_unlabeled.T
					dist = (uu + ll - 2*lu).sqrt()
					min_dist = torch.min(min_dist, dist.min(0)[0])

				cur_index = (min_dist - max_min_dist).argmax()
				max_min_dist = min_dist[cur_index]
				chosen_index = batch_size * batch_index + cur_index

			self.label_indices([chosen_index])


	def label_data(self, model):
		with torch.no_grad():
			if self.hparams.aquisition_method == 'random':
				self.label_randomly(self.hparams.batch_budget)
			elif self.hparams.aquisition_method == 'uncertain':
				self.label_uncertain(self.hparams.batch_budget, model)
			elif self.hparams.aquisition_method == 'learning-loss':
				self.label_highest_loss(self.hparams.batch_budget, model)
			elif self.hparams.aquisition_method == 'core-set':
				self.label_core_set(self.hparams.batch_budget, model)
			else:
				raise ValueError('Given aquisition method is not available')



class IALModel(pl.LightningModule):
	def __init__(self, 
		image_size: int,
		image_depth: int,
		layers_conv: list,
		layers_fc: list,
		classes: int,
		**kwargs
	):
		super().__init__()

		self.save_hyperparameters()
		
		# TODO Make these hyperparameters
		size_learning_loss = 16
		conv_stride = 3
		conv_pool = 2

		convolutional = []
		size_prev = image_depth
		final_size = image_size
		for size in layers_conv:
			convolutional += [
				torch.nn.Conv2d(size_prev, size, conv_stride),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(conv_pool, conv_pool)
			]
			final_size = (final_size - conv_stride + 1) // conv_pool
			size_prev = size
		convolutional.append(torch.nn.Flatten(1))
		self.convolutional = torch.nn.Sequential(*convolutional)

		self.fully_connected = torch.nn.ModuleList()
		size_prev = layers_conv[-1] * final_size**2
		for size in layers_fc:
			self.fully_connected.append(torch.nn.Sequential(torch.nn.Linear(size_prev, size), torch.nn.ReLU()))
			size_prev = size

		self.classifier = torch.nn.Linear(layers_fc[-1], 1 if classes == 2 else classes)

		if classes <= 2:
			self.loss = data_utils.bce_tofloat_loss
		else:
			self.loss = torch.nn.functional.cross_entropy

		self.accuracy = torchmetrics.Accuracy()

		if self.hparams.aquisition_method == 'learning-loss':
			self.loss_layers = [
				torch.nn.Sequential(torch.nn.Linear(size, size_learning_loss), torch.nn.ReLU())
				for size in layers_fc
			]

			self.loss_regressor = torch.nn.Linear(size_learning_loss * len(layers_fc), 1)


	def forward(self, images):
		pred_loss = None
		hidden = self.convolutional(images)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = []
			for step, layer in enumerate(self.fully_connected):
				hidden = layer(hidden)
				losses.append(self.loss_layers[step](hidden.detach()))

			pred_loss = self.loss_regressor(torch.cat(losses, 1)).squeeze()
		else:
			for layer in self.fully_connected:
				hidden = layer(hidden)

		preds = self.classifier(hidden).squeeze(1)

		return preds, pred_loss


	def training_step(self, batch, batch_idx):
		images, labels = batch
		labels_hat, losses_hat = self(images)

		loss = self.loss(labels_hat, labels)
		self.log("training classification loss", loss)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = self.loss(labels_hat, labels, reduction='none')
			loss_loss = data_utils.loss_loss(losses_hat, losses)
			self.log("training loss loss", loss_loss)

			loss += self.hparams.learning_loss_factor * loss_loss

		return loss

	def on_train_end(self):
		# https://github.com/PyTorchLightning/pytorch-lightning/issues/5007
		self.trainer.fit_loop.current_epoch += 1

		# To force skip early stopping the next epoch
		self.trainer.fit_loop.min_epochs = self.trainer.fit_loop.current_epoch + self.hparams.min_epochs


	def validation_step(self, batch, batch_idx):
		images, labels = batch
		labels_hat, losses_hat = self(images)

		loss = self.loss(labels_hat, labels)
		self.log("validation classification loss", loss)

		accuracy = self.accuracy(labels_hat, labels)
		self.log("validation classification accuracy", accuracy)

		num_labeled = float(len(self.trainer.datamodule.data_train.indices))
		self.log("labeled data", num_labeled)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = self.loss(labels_hat, labels, reduction='none')
			loss_loss = data_utils.loss_loss(losses_hat, losses)
			self.log("validation loss loss", loss_loss)

		return loss


	def test_step(self, batch, batch_idx):
		images, labels = batch
		labels_hat, losses_hat = self(images)

		loss = self.loss(labels_hat, labels)
		self.log("test classification loss", loss)

		accuracy = self.accuracy(labels_hat, labels)
		self.log("test classification accuracy", accuracy)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = self.loss(labels_hat, labels, reduction='none')
			loss_loss = data_utils.loss_loss(losses_hat, losses)
			self.log("test loss loss", loss_loss)

		return loss


	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
