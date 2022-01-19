#!/usr/bin/env python3

import torch
import pytorch_lightning as pl

import data_utils
import modules_mnist_binary
import modules_mnist
import modules_cifar10
import modules_svhn

from main import parse_arguments


def main():
	test_datasets()
	test_aquisition_methods()


def test_datasets():
	for dataset in ['mnist-binary', 'mnist', 'cifar10', 'svhn']:
		args = parse_arguments([
			dataset,
			'random'
		])

		trainer = pl.Trainer(
			logger=None,
			checkpoint_callback=False,
			max_epochs=1
		)

		if args.dataset == 'mnist-binary':
			model = modules_mnist_binary.MNISTBinaryModel(**vars(args))
			datamodule = modules_mnist_binary.MNISTBinaryDataModule(**vars(args))
		elif args.dataset == 'mnist':
			model = modules_mnist.MNISTModel(**vars(args))
			datamodule = modules_mnist.MNISTDataModule(**vars(args))
		elif args.dataset == 'cifar10':
			model = modules_cifar10.CIFAR10Model(**vars(args))
			datamodule = modules_cifar10.CIFAR10DataModule(**vars(args))
		elif args.dataset == 'svhn':
			model = modules_svhn.SVHNModel(**vars(args))
			datamodule = modules_svhn.SVHNDataModule(**vars(args))
		else:
			raise ValueError('Given dataset is not available')

		trainer.fit(model, datamodule)
		trainer.test(model, datamodule)


def test_aquisition_methods():
	for aquisition_method in ['random', 'uncertain', 'learning-loss', 'core-set']:
		args = parse_arguments([
			'mnist-binary',
			aquisition_method,
			'--batch-budget=10'
		])

		if args.dataset == 'mnist-binary':
			model = modules_mnist_binary.MNISTBinaryModel(**vars(args))
			datamodule = modules_mnist_binary.MNISTBinaryDataModule(**vars(args))
		elif args.dataset == 'mnist':
			model = modules_mnist.MNISTModel(**vars(args))
			datamodule = modules_mnist.MNISTDataModule(**vars(args))
		elif args.dataset == 'cifar10':
			model = modules_cifar10.CIFAR10Model(**vars(args))
			datamodule = modules_cifar10.CIFAR10DataModule(**vars(args))
		elif args.dataset == 'svhn':
			model = modules_svhn.SVHNModel(**vars(args))
			datamodule = modules_svhn.SVHNDataModule(**vars(args))
		else:
			raise ValueError('Given dataset is not available')

		datamodule.setup('fit')

		with torch.no_grad():
			if args.aquisition_method == 'random':
				data_utils.label_randomly(datamodule, args.batch_budget)
			elif args.aquisition_method == 'uncertain':
				data_utils.label_uncertain(datamodule, args.batch_budget, model)
			elif args.aquisition_method == 'learning-loss':
				data_utils.label_highest_loss(datamodule, args.batch_budget, model)
			elif args.aquisition_method == 'core-set':
				data_utils.label_core_set(datamodule, args.batch_budget, model)
			else:
				raise ValueError('Given aquisition method is not available')
		
		assert \
			len(datamodule.data_train.indices) == len(set(datamodule.data_train.indices)), \
			'Duplicate labeled data points detected'


if __name__ == "__main__":
	main()
