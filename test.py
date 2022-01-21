#!/usr/bin/env python3

import pytorch_lightning as pl

import data_utils
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
		model, datamodule = data_utils.get_modules(args)

		trainer.fit(model, datamodule)
		trainer.test(model, datamodule)


def test_aquisition_methods():
	for aquisition_method in ['random', 'uncertain', 'learning-loss', 'core-set']:
		args = parse_arguments([
			'mnist-binary',
			aquisition_method,
			'--batch-budget=10'
		])

		model, datamodule = data_utils.get_modules(args)
		datamodule.setup('fit')

		datamodule.label_data(model)

		assert \
			len(datamodule.data_train.indices) == len(set(datamodule.data_train.indices)), \
			'Duplicate labeled data points detected'


if __name__ == "__main__":
	main()
