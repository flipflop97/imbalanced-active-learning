import random
import torch


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
		model = modules_mnist_binary.MNISTBinaryModel(**vars(args))
		datamodule = modules_mnist_binary.MNISTBinaryDataModule(**vars(args))
	elif args.dataset == 'mnist':
		import modules_mnist
		model = modules_mnist.MNISTModel(**vars(args))
		datamodule = modules_mnist.MNISTDataModule(**vars(args))
	elif args.dataset == 'cifar10':
		import modules_cifar10
		model = modules_cifar10.CIFAR10Model(**vars(args))
		datamodule = modules_cifar10.CIFAR10DataModule(**vars(args))
	elif args.dataset == 'svhn':
		import modules_svhn
		model = modules_svhn.SVHNModel(**vars(args))
		datamodule = modules_svhn.SVHNDataModule(**vars(args))
	else:
		raise ValueError('Given dataset is not available')

	return model, datamodule
