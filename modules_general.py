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

		self.setup_fit_done = False
		self.setup_test_done = False


	def setup(self, stage:str=None):
		if stage in ["fit", "validate", None] and not self.setup_fit_done:
			self.setup_fit_done = True

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
			self.label_static_distribution(self.hparams.initial_labels)

		if stage in ["test", None] and not self.setup_test_done:
			self.setup_test_done = True

			self.data_test = self.get_data_test()


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

	def labeled_dataloader_single(self):
		return torch.utils.data.DataLoader(
			self.data_train,
			batch_size=1,
			shuffle=True,
			num_workers=self.hparams.dataloader_workers
		)

	def unlabeled_dataloader(self):
		return torch.utils.data.DataLoader(
			self.data_unlabeled,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.dataloader_workers
		)

	def unlabeled_dataloader_single(self):
		return torch.utils.data.DataLoader(
			self.data_unlabeled,
			batch_size=1,
			num_workers=self.hparams.dataloader_workers
		)


	@property
	def class_balance(self):
		return torch.tensor([len([index
				for index in self.data_train.indices
				if self.data_train.dataset.targets[index] == c
			]) for c, _ in enumerate(self.data_train.dataset.classes)
		])


	def label_indices(self, indices: list):
		self.data_train.indices = sorted(list(set(self.data_train.indices + indices)))
		self.data_unlabeled.indices = sorted(list(set([index
			for index in self.data_unlabeled.indices
			if index not in indices
		])))


	def label_each_class(self, amount: int = 1):
		self.label_indices(sum([
			random.sample([index
				for index in self.data_unlabeled.indices
				if self.data_unlabeled.dataset.targets[index] == class_num
			], amount)
			for class_num, _ in enumerate(self.data_unlabeled.dataset.classes)
		], []))


	def label_static_distribution(self, amount: int):
		'''
		Label images randomly following the distribution of unlabeled data
		'''
		unlabeled_data_count = len(self.data_unlabeled)
		unlabeled_class_indices = [
			[index
				for index in self.data_unlabeled.indices
				if self.data_unlabeled.dataset.targets[index] == class_num
			] for class_num, _ in enumerate(self.data_unlabeled.dataset.classes)
		]

		# Label amount of images from each class relative to its size (rounded down)
		labels_remaining = amount
		for class_indices in unlabeled_class_indices:
			class_labeling_count = int(amount * len(class_indices) / unlabeled_data_count)
			self.label_indices(random.sample(class_indices, class_labeling_count))
			labels_remaining -= class_labeling_count

		# Choose random classes to label the remaining images
		self.label_indices(sum([
			random.sample([index
				for index in self.data_unlabeled.indices
				if self.data_unlabeled.dataset.targets[index] == class_num
			], 1)
			for class_num in random.sample(range(len(self.data_unlabeled.dataset.classes)), labels_remaining)
		], []))


	def label_randomly(self, amount: int, model: pl.LightningModule = None):
		chosen_indices = random.sample(self.data_unlabeled.indices, amount)
		self.label_indices(chosen_indices)


	def label_uncertain(self, amount: int, model: pl.LightningModule, uncertainty_method: str):
		if uncertainty_method == 'entropy':
			def uncertainty_method_fn(preds):
				return -(preds*preds.log()).sum(1).nan_to_num(0)

		elif uncertainty_method == 'margin':
			def uncertainty_method_fn(preds):
				return 1 - preds.topk(2, dim=1)[0].diff(dim=1).abs().squeeze(1)

		elif uncertainty_method == 'least-confident':
			def uncertainty_method_fn(preds):
				return 1 - preds.max(1)[0]

		else:
			raise ValueError(f"{uncertainty_method} is no valid uncertainty method")

		uncertainty_list = []
		with torch.no_grad():
			for batch in tqdm.tqdm(self.unlabeled_dataloader(), desc='Labeling'):
				images, _ = batch
				output, _ = model(images)

				try:
					# Multiclass, softmax
					preds = torch.nn.functional.softmax(output, 1)
				except IndexError:
					# Binary, sigmoid
					preds_binary = torch.sigmoid(output)
					preds = torch.stack([preds_binary, 1 - preds_binary], 1)

				uncertainty_list.append(uncertainty_method_fn(preds))

			uncertainty = torch.cat(uncertainty_list)
			_, top_indices = uncertainty.topk(amount)
			chosen_indices = [self.data_unlabeled.indices[i] for i in top_indices]
			self.label_indices(chosen_indices)

	def label_entropy(self, amount: int, model: pl.LightningModule):
		self.label_uncertain(amount, model, 'entropy')

	def label_margin(self, amount: int, model: pl.LightningModule):
		self.label_uncertain(amount, model, 'margin')

	def label_least_confident(self, amount: int, model: pl.LightningModule):
		self.label_uncertain(amount, model, 'least-confident')


	# Active Learning for Skewed Data Sets
	# Abbas Kazerouni et al
	def label_hal_r(self, amount: int, model: pl.LightningModule):
		dist = numpy.random.binomial(amount, self.hparams.hal_exploit_probability)

		self.label_margin(dist, model)
		self.label_randomly(amount - dist, model)

	def label_hal_g(self, amount: int, model: pl.LightningModule):
		dist = numpy.random.binomial(amount, self.hparams.hal_exploit_probability)

		self.label_margin(dist, model)

		batch_size = self.hparams.eval_batch_size

		with torch.no_grad():
			for _ in tqdm.trange(amount - dist, desc='Labeling'):
				min_sum_dist = 0
				cache_labeled = []

				for batch_labeled in self.labeled_dataloader():
					x_labeled, _ = batch_labeled
					h_labeled = model.convolutional(x_labeled)
					for layer in model.fully_connected:
						h_labeled = layer(h_labeled)
					cache_labeled.append(h_labeled)

				for batch_index, batch_unlabeled in enumerate(self.unlabeled_dataloader()):
					x_unlabeled, _ = batch_unlabeled
					h_unlabeled = model.convolutional(x_unlabeled)
					for layer in model.fully_connected:
						h_unlabeled = layer(h_unlabeled)

					sum_dist = torch.full([len(h_unlabeled)], numpy.Inf)
					for h_labeled in cache_labeled:
						uu = h_unlabeled.pow(2).sum(1, keepdim=True).T
						ll = h_labeled.pow(2).sum(1, keepdim=True)
						lu = h_labeled @ h_unlabeled.T
						dist = (uu + ll - 2*lu).sqrt()
						dist_gauss = torch.exp(- dist / self.hparams.hal_gaussian_variance)
						sum_dist = sum_dist + dist_gauss.sum(0)

					cur_index = (sum_dist - min_sum_dist).argmin()
					min_sum_dist = sum_dist[cur_index]
					chosen_index = self.data_unlabeled.indices[batch_size * batch_index + cur_index]

				self.label_indices([chosen_index])


	# Class-Balanced Active Learning for Image Classification
	# Javad Zolfaghari Bengar, Joost van de Weijer, Laura Lopez Fuentes, Bogdan Raducanu
	def label_class_balanced(self, amount: int, model: pl.LightningModule):
		raise NotImplementedError

	def label_class_balanced_greedy(self, amount: int, model: pl.LightningModule):
		# Greedy: Essentially same as uncertain except:
		#   - Add to uncertainty values: lambda * (max(0, labeled/classes - labeled_class) - expected_classes)
		#   - Label points one at a time

		batch_size = self.hparams.eval_batch_size

		with torch.no_grad():
			for _ in tqdm.trange(amount, desc='Labeling'):
				max_uncertainty = float('-inf')
				for batch_index, batch in enumerate(self.unlabeled_dataloader()):
					images, _ = batch
					output, _ = model(images)

					try:
						# Multiclass, softmax
						preds = torch.nn.functional.softmax(output, 1)
					except IndexError:
						# Binary, sigmoid
						preds_binary = torch.sigmoid(output)
						preds = torch.stack([preds_binary, 1 - preds_binary], 1)

					uncertainty_score = -(preds*preds.log()).sum(1).nan_to_num(0)
					balance_omega = torch.clamp(len(self.data_train) / len(self.data_train.dataset.classes) - self.class_balance, min=0)
					balance_penalty = self.hparams.class_balancing_factor * torch.norm(balance_omega.unsqueeze(0) - preds, p=1, dim=1)

					cur_uncertainty, cur_index = torch.max(uncertainty_score - balance_penalty, axis=0)
					if cur_uncertainty > max_uncertainty:
						max_index = batch_size * batch_index + cur_index
						max_uncertainty = cur_uncertainty

				chosen_index = self.data_unlabeled.indices[max_index]
				self.label_indices([chosen_index])


	# Learning Loss for Active Learning
	# Donggeun Yoo, In So Kweon
	def label_highest_loss(self, amount: int, model: pl.LightningModule):
		with torch.no_grad():
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
	def label_k_center(self, amount: int, model: pl.LightningModule):
		with torch.no_grad():
			raise NotImplementedError

	def label_k_center_greedy(self, amount: int, model: pl.LightningModule):
		# Each time, get the unlabeled data point with the largest minimum distance to a labeled data point
		batch_size = self.hparams.eval_batch_size

		with torch.no_grad():
			for _ in tqdm.trange(amount, desc='Labeling'):
				max_min_dist = 0
				cache_labeled = []

				for batch_labeled in self.labeled_dataloader():
					x_labeled, _ = batch_labeled
					h_labeled = model.convolutional(x_labeled)
					for layer in model.fully_connected:
						h_labeled = layer(h_labeled)
					cache_labeled.append(h_labeled)

				for batch_index, batch_unlabeled in enumerate(self.unlabeled_dataloader()):
					x_unlabeled, _ = batch_unlabeled
					h_unlabeled = model.convolutional(x_unlabeled)
					for layer in model.fully_connected:
						h_unlabeled = layer(h_unlabeled)

					min_dist = torch.full([len(h_unlabeled)], numpy.Inf)
					for h_labeled in cache_labeled:
						uu = h_unlabeled.pow(2).sum(1, keepdim=True).T
						ll = h_labeled.pow(2).sum(1, keepdim=True)
						lu = h_labeled @ h_unlabeled.T
						dist = (uu + ll - 2*lu)#.sqrt()
						min_dist = torch.min(min_dist, dist.min(0)[0])

					cur_index = (min_dist - max_min_dist).argmax()
					max_min_dist = min_dist[cur_index]
					chosen_index = self.data_unlabeled.indices[batch_size * batch_index + cur_index]

				self.label_indices([chosen_index])


	# https://github.com/nimarb/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_function.py
	def rank_influence(self, model: pl.LightningModule):
		params = [p for p in model.parameters() if p.requires_grad]

		def calc_hvp(loss, s_test):
			first_grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
			elemwise_products = sum(torch.sum(grad_i * s_test_i) for grad_i, s_test_i in zip(first_grads, s_test))
			gradients = torch.autograd.grad(elemwise_products, params, create_graph=True, retain_graph=False)
			return [gradient.detach() for gradient in gradients]

		def calc_v():
			loss = 0
			for images, targets in self.val_dataloader():
				predictions, _ = model(images)
				loss += model.loss(predictions, targets, reduction='sum')
			loss /= len(self.data_val)

			gradients = torch.autograd.grad(loss, params, create_graph=True, retain_graph=False)
			return [gradient.detach() for gradient in gradients]

		def calc_s_test():
			v = calc_v()
			s_test = v.copy()

			current_iteration = 0
			while current_iteration < self.hparams.influence_max_iterations:
				for images, targets in self.labeled_dataloader_single():
					predictions, _ = model(images)
					loss = model.loss(predictions, targets)
					hvp = calc_hvp(loss, s_test)

					# DEBUGGING STUFF
					print("cur_it", current_iteration)
					print("     v", v[-1][0].item())
					print("   hvp", hvp[-1][0].item())
					print("s_test", s_test[-1][0].item())

					s_test = [(v_i + s_test_i - hvp_i).detach() for v_i, s_test_i, hvp_i in zip(v, s_test, hvp)]

					# DEBUGGING STUFF
					print("new_st", s_test[-1][0].item())
					print()
					for unit in hvp:
						if unit.isnan().any():
							raise ValueError("One or more values of s_test bacame NaN")

					current_iteration += 1
					if current_iteration >= self.hparams.influence_max_iterations:
						break

			return s_test

		s_test = calc_s_test()

		# TODO Is this batchable?
		influences = []
		for images, _ in tqdm.tqdm(self.unlabeled_dataloader_single(), desc='Labeling'):
			predictions, _ = model(images)
			certainties, targets = model.guess(predictions)
			loss = model.loss(predictions, targets)
			g_z = [gradients.detach() * certainties for gradients in torch.autograd.grad(loss, params, create_graph=True, retain_graph=False)]
			influence = -sum(float(torch.sum(s_test_i * g_z_i)) for s_test_i, g_z_i in zip(s_test, g_z))
			influences.append(influence)

		return torch.tensor(influences)

	def label_influence(self, amount: int, model: pl.LightningModule):
		influences = self.rank_influence(model)

		_, top_indices = influences.topk(amount)
		chosen_indices = [self.data_unlabeled.indices[i] for i in top_indices]
		self.label_indices(chosen_indices)

	def label_influence_abs(self, amount: int, model: pl.LightningModule):
		influences = self.rank_influence(model).abs()

		_, top_indices = influences.topk(amount)
		chosen_indices = [self.data_unlabeled.indices[i] for i in top_indices]
		self.label_indices(chosen_indices)

	def label_influence_neg(self, amount: int, model: pl.LightningModule):
		influences = -self.rank_influence(model)

		_, top_indices = influences.topk(amount)
		chosen_indices = [self.data_unlabeled.indices[i] for i in top_indices]
		self.label_indices(chosen_indices)


	def label_data(self, model):
		aquisition_methods = {
			'random': self.label_randomly,
			'least-confident': self.label_least_confident,
			'margin': self.label_margin,
			'entropy': self.label_entropy,
			'learning-loss': self.label_highest_loss,
			'k-center': self.label_k_center,
			'k-center-greedy': self.label_k_center_greedy,
			'class-balanced': self.label_class_balanced,
			'class-balanced-greedy': self.label_class_balanced_greedy,
			'hal-r': self.label_hal_r,
			'hal-g': self.label_hal_g,
			'influence': self.label_influence,
			'influence-abs': self.label_influence_abs,
			'influence-neg': self.label_influence_neg,
		}

		cb_before = self.class_balance / len(self.data_train) * 100
		aquisition_methods[self.hparams.aquisition_method](self.hparams.labeling_budget, model)
		cb_after = self.class_balance / len(self.data_train) * 100

		print('Data labeled, class balance:')

		max_class_len = max(len(cls) for cls in self.data_train.dataset.classes)
		for num, cls in enumerate(self.data_train.dataset.classes):
			print(f"{cls:{max_class_len}}  {cb_before[num]:2.0f}% -> {cb_after[num]:2.0f}% ({cb_after[num] - cb_before[num]:+2.0f}%)")



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

		self.example_input_array = torch.zeros([self.hparams.train_batch_size, image_depth, image_size, image_size])

		convolutional = []
		size_prev = image_depth
		final_size = image_size
		for size in layers_conv:
			convolutional += [
				torch.nn.Conv2d(size_prev, size, self.hparams.convolutional_stride),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(self.hparams.convolutional_pool, self.hparams.convolutional_pool)
			]
			final_size = (final_size - self.hparams.convolutional_stride + 1) // self.hparams.convolutional_pool
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
			self.binary = True
			self.loss = data_utils.bce_tofloat_loss
		else:
			self.binary = False
			self.loss = torch.nn.functional.cross_entropy

		self.accuracy = torchmetrics.Accuracy()

		if self.hparams.aquisition_method == 'learning-loss':
			self.loss_layers = torch.nn.ModuleList([
				torch.nn.Sequential(torch.nn.Linear(size, self.hparams.learning_loss_layer_size), torch.nn.ReLU())
				for size in layers_fc
			])

			self.loss_regressor = torch.nn.Linear(self.hparams.learning_loss_layer_size * len(layers_fc), 1)


	def forward(self, images):
		pred_loss = torch.empty(0)
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
		self.log("running/classification/training/loss", loss, on_step=False, on_epoch=True)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = self.loss(labels_hat, labels, reduction='none')
			loss_loss = data_utils.loss_loss(losses_hat, losses)
			self.log("running/learning-loss/training/loss", loss_loss, on_step=False, on_epoch=True)

			loss += self.hparams.learning_loss_factor * loss_loss

		return loss

	def on_train_end(self):
		# To force skip early stopping the next epoch
		self.trainer.fit_loop.min_epochs = self.trainer.fit_loop.epoch_progress.current.processed + self.hparams.min_epochs


	def validation_step(self, batch, batch_idx):
		images, labels = batch
		labels_hat, losses_hat = self(images)

		loss = self.loss(labels_hat, labels)
		self.log("running/classification/validation/loss", loss)

		accuracy = self.accuracy(labels_hat, labels)
		self.log("running/classification/validation/accuracy", accuracy)

		num_labeled = float(len(self.trainer.datamodule.data_train.indices))
		self.log("running/labeled-data/count", num_labeled)

		class_balance = self.trainer.datamodule.class_balance / len(self.trainer.datamodule.data_train)
		entropy_labeled = -(class_balance * class_balance.log()).sum()
		self.log("running/labeled-data/entropy", entropy_labeled)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = self.loss(labels_hat, labels, reduction='none')
			loss_loss = data_utils.loss_loss(losses_hat, losses)
			self.log("running/learning-loss/validation/loss", loss_loss)

		return loss


	def test_step(self, batch, batch_idx):
		images, labels = batch
		labels_hat, losses_hat = self(images)

		loss = self.loss(labels_hat, labels)
		self.log("running/classification/test/loss", loss)

		accuracy = self.accuracy(labels_hat, labels)
		self.log("running/classification/test/accuracy", accuracy)

		if self.hparams.aquisition_method == 'learning-loss':
			losses = self.loss(labels_hat, labels, reduction='none')
			loss_loss = data_utils.loss_loss(losses_hat, losses)
			self.log("running/learning-loss/test/loss", loss_loss)

		return loss


	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


	def guess(self, predictions):
		if self.binary:
			certainties = predictions
			targets = (predictions > 0).int()
		else:
			certainties, targets = predictions.max(1)

		return certainties, targets
