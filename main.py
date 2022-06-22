#!/usr/bin/env python3

import collections
import argparse
import torch
import pytorch_lightning as pl

import data_utils


def parse_arguments(*args, **kwargs):
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	def percentage(arg):
		try:
			f = float(arg)
		except ValueError:
			raise argparse.ArgumentTypeError(f"invalid float value: '{arg}'")
		if f < 0 or f > 1:
			raise argparse.ArgumentTypeError(f"value is not between 0 and 1: '{arg}'")
		return f

	# Model related
	parser.add_argument(
		'dataset', type=str,
		choices=['mnist-binary', 'mnist', 'cifar10', 'svhn'],
		help="The dataset and corresponding model"
	)
	parser.add_argument(
		'--train-split', type=percentage, default=0.8,
		help="Percentage of the data to be used for training, the rest will be used for validation"
	)
	parser.add_argument(
		'--learning-rate', type=float, default=2e-4,
		help="Multiplier used to tweak model parameters"
	)
	parser.add_argument(
		'--train-batch-size', type=int, default=8,
		help="Batch size used for training the model"
	)
	parser.add_argument(
		'--min-epochs', type=int, default=50,
		help="Minimum epochs to train before switching to the early stopper"
	)
	parser.add_argument(
		'--convolutional-stride', type=int, default=3,
		help="Stride used by convolutional layers"
	)
	parser.add_argument(
		'--convolutional-pool', type=int, default=2,
		help="Max pooling used by convolutional layers"
	)
	parser.add_argument(
		'--seed', type=int, default=None,
		help="Max pooling used by convolutional layers"
	)

	# Active learning related
	parser.add_argument(
		'aquisition_method', type=str,
		choices=[
			'random',
			'least-confident', 'margin', 'entropy',
			'learning-loss',
			'k-center-greedy',
			'class-balanced-greedy',
			'hal-r', 'hal-g',
			'influence', 'influence-abs', 'influence-neg', 'influence-real',
		],
		help="The unlabeled data aquisition method to use"
	)
	parser.add_argument(
		'--early-stopping-patience', type=int, default=10,
		help="Epochs to wait before stopping training and asking for new data"
	)
	parser.add_argument(
		'--class-balance', type=float, default=0.5,
		help="Class balance multiplier for half of the classes"
	)
	parser.add_argument(
		'--initial-labels', type=int, default=100,
		help="The amount of initially labeled datapoints"
	)
	parser.add_argument(
		'--labeling-budget', type=int, default=50,
		help="The amount of datapoints to be labeled per aquisition step"
	)
	parser.add_argument(
		'--labeling-steps', type=int, default=10,
		help="The total amount of aquisition steps"
	)
	parser.add_argument(
		'--learning-loss-factor', type=float, default=0.1,
		help="Multiplier used on top of the learning rate for the additional learning loss"
	)
	parser.add_argument(
		'--learning-loss-layer-size', type=int, default=16,
		help="Layer size used by learning loss layers"
	)
	parser.add_argument(
		'--class-balancing-factor', type=float, default=1,
		help="Multiplier used for adjusting the class-balancing effect"
	)
	parser.add_argument(
		'--influence-max-iterations', type=int, default=5000,
		help="Maximum iterations of calculating s_test for influence aquisition"
	)
	parser.add_argument(
		'--hal-exploit-probability', type=percentage, default=0.5,
		help="Probability in HAL of choosing the exploit aquisition as opposed to the exploring one"
	)
	parser.add_argument(
		'--hal-gaussian-variance', type=float, default=10,
		help="Variance used by HAL-G aquisition in calculating gaussian distance"
	)
	parser.add_argument(
		'--influence-damp', type=percentage, default=0.01,
		help="Damping factor used for calculating s_test in influence calculation"
	)
	parser.add_argument(
		'--influence-scale', type=percentage, default=1e5,
		help="Scaling factor used for calculating s_test in influence calculation"
	)

	# Device related
	parser.add_argument(
		'--data-dir', type=str, default='./datasets',
		help="Multiplier used to tweak model parameters"
	)
	parser.add_argument(
		'--eval-batch-size', type=int, default=8192,
		help="Batch size used for evaluating the model"
	)
	parser.add_argument(
		'--dataloader-workers', type=int, default=4,
		help="Amount of workers used for dataloaders"
	)
	parser.add_argument(
		'--disable-logging', action='store_true',
		help="Disable logging results to WandB"
	)

	return parser.parse_args(*args, **kwargs)


def reset_weights(layer):
	if hasattr(layer, 'reset_parameters'):
		layer.reset_parameters()


def main():
	try:
		args = parse_arguments()
		pl.seed_everything(args.seed, workers=True)

		if args.disable_logging:
			logger = pl.loggers.base.DummyLogger()
		else:
			logger = pl.loggers.WandbLogger(name=f"{args.dataset} {args.aquisition_method} {args.class_balance}")

		early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
			monitor='running/classification/validation/loss',
			mode='min',
			patience=args.early_stopping_patience
		)
		use_gpu = torch.cuda.is_available()
		trainer = pl.Trainer(
			gpus=int(use_gpu),
			auto_select_gpus=use_gpu,
			deterministic=(args.seed is not None),
			log_every_n_steps=10,
			min_epochs=args.min_epochs,
			max_epochs=-1,
			logger=logger,
			callbacks=[early_stopping_callback]
		)
		model, datamodule = data_utils.get_modules(args)

		trainer.validate(model, datamodule)
		if trainer.interrupted:
			raise KeyboardInterrupt
		trainer.test(model, datamodule)
		if trainer.interrupted:
			raise KeyboardInterrupt

		auc_logs = collections.Counter()
		for step in range(args.labeling_steps):
			model.apply(reset_weights)
			ial_logs = dict()

			trainer.fit(model, datamodule)
			if trainer.interrupted:
				raise KeyboardInterrupt
			ial_logs.update({label.replace("running/", "final/"): value for label, value in trainer.logged_metrics.items()})

			trainer.test(model, datamodule)
			if trainer.interrupted:
				raise KeyboardInterrupt
			ial_logs.update({label.replace("running/", "final/"): value for label, value in trainer.logged_metrics.items()})

			logger.log_metrics(ial_logs)
			auc_logs += ial_logs

			early_stopping_callback.best_score = torch.tensor(float('inf'))

			if step < args.labeling_steps - 1:
				datamodule.label_data(model)

		auc_logs = {key.replace("final/", "auc/"): val for key, val in auc_logs.items()}
		logger.log_metrics(auc_logs)

	except KeyboardInterrupt:
		pass


if __name__ == "__main__":
	main()
