# Imbalanced Active Learning

Supplementary code for my thesis on active learning for imbalanced data sets. This framework contains code to run and compare different active learning methods on different datasets to be able to compare them.

## Environment

For replicating experiments with the originally used library versions, import the stable environment:

```bash
$ conda env create -f conda_env_stable.yml
```

For continuation of work on this framework with the newest library versions, import the rolling environment:

```bash
$ conda env create -f conda_env_rolling.yml
```

## Running

Experiments can be executed using `main.py`, datasets will be downloaded automatically. Parameters can be set via command line parameters, check the help for available parameters:

```bash
$ ./main.py --help
```
