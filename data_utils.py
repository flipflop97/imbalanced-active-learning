import random
import torch

import modules_general


def loss_loss(
	input: torch.Tensor,
	target: torch.Tensor,
	margin: float = 1.0,
	reduction: str = 'mean'
) -> torch.Tensor:
	halfway = len(input)//2
	indicator = (target - target.flip(0))[:halfway].sign()
	pair_diff = (input - input.flip(0))[:halfway]
	loss = torch.clamp(-indicator * pair_diff + margin, min=0)

	if reduction == 'none':
		return loss
	elif reduction == 'mean':
		return loss.mean()
	elif reduction == 'sum':
		return loss.sum()


def bce_tofloat_loss(pred, target, *args, **kwargs):
	return torch.nn.functional.binary_cross_entropy_with_logits(pred, target.float(), *args, **kwargs)


def balance_classes(subset: torch.utils.data.Subset, balance_factor: float):
	# Divide all indices into classes
	class_indices = [
		[index for index in subset.indices if subset.dataset.targets[index] == c]
		for c, _ in enumerate(subset.dataset.classes)
	]

	# Generate class balance list
	head = len(class_indices) // 2
	tail = len(class_indices) - head
	balance = [balance_factor]*head + [1.0]*tail

	# Blance class_indices by the class balance weights
	ref = min(len(indices) / balance[c] for c, indices in enumerate(class_indices))
	balanced_indices = [random.sample(indices, int(ref * balance[c]))
        for c, indices in enumerate(class_indices)
    ]

	subset.indices = sum(balanced_indices, [])


def get_modules(args):
	if args.dataset == 'mnist-binary':
		import modules_mnist_binary
		datamodule = modules_mnist_binary.MNISTBinaryDataModule(**vars(args))
		model = modules_general.IALModel(28, 1, [6, 16], [128, 64], 2, **vars(args))
	elif args.dataset == 'mnist':
		import modules_mnist
		datamodule = modules_mnist.MNISTDataModule(**vars(args))
		model = modules_general.IALModel(28, 1, [6, 16], [128, 64], 10, **vars(args))
	elif args.dataset == 'cifar10':
		import modules_cifar10
		datamodule = modules_cifar10.CIFAR10DataModule(**vars(args))
		model = modules_general.IALModel(32, 3, [12, 16, 20], [128, 64], 10, **vars(args))
	elif args.dataset == 'svhn':
		import modules_svhn
		datamodule = modules_svhn.SVHNDataModule(**vars(args))
		model = modules_general.IALModel(32, 3, [12, 16, 20], [128, 64], 10, **vars(args))
	else:
		raise ValueError('Given dataset is not available')

	return model, datamodule
