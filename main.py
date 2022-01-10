#!/usr/bin/env python3

import argparse
import numpy
import torch
import pytorch_lightning as pl

import data_utils
import modules_mnist


def parse_arguments(*args, **kwargs):
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
		'--eval-batch-size', type=int, default=1024,
		help="Batch size used for evaluating the model"
	)
	parser.add_argument(
		'--dataloader-workers', type=int, default=4,
		help="Amount of workers used for dataloaders"
	)

	return parser.parse_args(*args, **kwargs)


def main():
	args = parse_arguments()

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
	model = modules_mnist.ALModel28(**vars(args))
	mnist = modules_mnist.MNISTDataModule(**vars(args))

	# TODO Think of a more appropriate limit
	for _ in range(20):
		trainer.fit(model, mnist)
		trainer.test(model, mnist)

		# TODO Could this be moved to on_train_end?
		early_stopping_callback.best_score = torch.tensor(numpy.Inf)

		# TODO Would it be possible to do this in a callback?
		with torch.no_grad():
			if args.aquisition_method == 'random':
				data_utils.label_randomly(mnist, args.batch_budget)
			elif args.aquisition_method == 'uncertain':
				data_utils.label_uncertain(mnist, args.batch_budget, model)
			elif args.aquisition_method == 'learning-loss':
				data_utils.label_highest_loss(mnist, args.batch_budget, model)
			elif args.aquisition_method == 'core-set':
				data_utils.label_core_set(mnist, args.batch_budget, model)
			else:
				raise ValueError('Given aquisition method is not available')


if __name__ == "__main__":
	main()
