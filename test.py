#!/usr/bin/env python3

import torch
import pytorch_lightning as pl

import data_utils
from main import parse_arguments


TEXT_DEFAULT    = '\033[0m'
TEXT_BOLD       = '\033[1m'
TEXT_DIM        = '\033[2m'
TEXT_ITALICS    = '\033[3m'
TEXT_UNDERLINED = '\033[4m'


def main():
	test_datasets()
	test_aquisition_methods()


def test_datasets():
	for dataset in ['mnist-binary', 'mnist', 'cifar10', 'svhn']:
		print(f"\n{TEXT_BOLD}Testing dataset {dataset}{TEXT_DEFAULT}")

		args = parse_arguments([
			dataset,
			'random'
		])

		trainer = pl.Trainer(
			gpus=list(range(torch.cuda.device_count())),
			logger=None,
			enable_checkpointing=False,
			max_epochs=1
		)
		model, datamodule = data_utils.get_modules(args)

		trainer.fit(model, datamodule)
		trainer.test(model, datamodule)


def test_aquisition_methods():
	for aquisition_method in ['random', 'uncertain', 'learning-loss', 'core-set']:
		print(f"\n{TEXT_BOLD}Testing aquisition method {aquisition_method}{TEXT_DEFAULT}")

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
