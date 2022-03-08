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
	try:
		test_datasets()
		test_aquisition_methods()
	except KeyboardInterrupt:
		pass


def test_datasets():
	for dataset in ['mnist-binary', 'mnist', 'cifar10', 'svhn']:
		print(f"\n{TEXT_BOLD}Testing dataset {dataset}{TEXT_DEFAULT}")

		args = parse_arguments([
			dataset,
			'random'
		])
		use_gpu = torch.cuda.is_available()

		trainer = pl.Trainer(
			gpus=int(use_gpu),
			auto_select_gpus=use_gpu,
			logger=None,
			enable_checkpointing=False,
			max_epochs=1
		)
		model, datamodule = data_utils.get_modules(args)

		trainer.fit(model, datamodule)
		if trainer.interrupted:
			raise KeyboardInterrupt

		trainer.test(model, datamodule)
		if trainer.interrupted:
			raise KeyboardInterrupt


def test_aquisition_methods():
	for aquisition_method in [
			'random',
			'uncertainty',
			'uncertainty-balanced-greedy',
			'learning-loss',
			'core-set-greedy',
			'hal-r'
		]:
		print(f"\n{TEXT_BOLD}Testing aquisition method {aquisition_method}{TEXT_DEFAULT}")

		args = parse_arguments([
			'mnist',
			aquisition_method,
			'--labeling-budget=10'
		])

		model, datamodule = data_utils.get_modules(args)
		datamodule.setup('fit')

		datamodule.label_data(model)


if __name__ == "__main__":
	main()
