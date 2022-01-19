import random
import numpy
import torch
import pytorch_lightning as pl


def balance_classes(subset: torch.utils.data.Subset, balance_factor: float):
	class_indices = [
		[index for index in subset.indices if subset.dataset.targets[index] == c]
		for c, _ in enumerate(subset.dataset.classes)
	]

	head = len(class_indices) // 2
	tail = len(class_indices) - head
	balance = [balance_factor]*head + [1.0]*tail

	ref = min(len(indices) / balance[c] for c, indices in enumerate(class_indices))
	balanced_indices = [random.sample(indices, int(ref * balance[c]))
        for c, indices in enumerate(class_indices)
    ]
	subset.indices = sum(balanced_indices, [])


def label_indices(datamodule: pl.LightningDataModule, indices: list):
	datamodule.data_train.indices += indices
	datamodule.data_unlabeled.indices = [index
        for index in datamodule.data_unlabeled.indices
        if index not in indices
    ]


def label_randomly(datamodule: pl.LightningDataModule, amount: int):
	chosen_indices = random.sample(datamodule.data_unlabeled.indices, amount)
	label_indices(datamodule, chosen_indices)


def label_uncertain(datamodule: pl.LightningDataModule, amount: int, model: pl.LightningModule):
	uncertainty_list = []
	for batch in datamodule.unlabeled_dataloader():
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
	chosen_indices = [datamodule.data_unlabeled.indices[i] for i in top_indices]
	label_indices(datamodule, chosen_indices)


def label_highest_loss(datamodule: pl.LightningDataModule, amount: int, model: pl.LightningModule):
	uncertainty_list = []
	for batch in datamodule.unlabeled_dataloader():
		x, _ = batch
		_, losses_hat = model(x)
		uncertainty_list.append(losses_hat)

	uncertainty = torch.cat(uncertainty_list)
	_, top_indices = uncertainty.topk(amount)
	chosen_indices = [datamodule.data_unlabeled.indices[i] for i in top_indices]
	label_indices(datamodule, chosen_indices)


def label_core_set(datamodule: pl.LightningDataModule, amount: int, model: pl.LightningModule):
	# Each time, get the unlabeled data point with the largest minimum distance to a labeled data point

	batch_size = datamodule.hparams.eval_batch_size

	for i in range(amount):
		max_min_dist = 0
		cache_labeled = []

		print(f"Label {i}", end="\r")

		for batch_labeled in datamodule.labeled_dataloader():
			x_labeled, _ = batch_labeled
			cache_labeled.append(model.convolutional(x_labeled))

		for batch_index, batch_unlabeled in enumerate(datamodule.unlabeled_dataloader()):
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

		label_indices(datamodule, [chosen_index])
