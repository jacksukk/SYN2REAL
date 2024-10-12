# Task Arithmetic can Mitigate Synthetic-to-Real Gap in Automatic Speech Recognition

**EMNLP 2024 Main**

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2406.02925)

Hsuan Su, Hua Farn, Fan-Yun Sun, Shang-Tse Chen, Hung-yi Lee



### Environmental Setup
Run the following commands to build up the environment

```
$ conda env create -f environment.yml
```

### Synthetic Data Generation
Synthesize speech given text from SLURP dataset with BARK model

```
$ source slurm_scripts/generate.sh
```

### ASR training
#### Train pretrained ASR models with source/target domain real and synthetic data

```
$ source slurm_scripts/train.sh
```
#### Train pretrained ASR models with source domain (i.e. source domain ASR, mixture of real and synthetic data)

```
$ source slurm_scripts/train_mix.sh
```

#### Train above source domain ASR models with target domain synthetic data (second stage)

```
$ source slurm_scripts/train_mix_continue.sh
```

### ASR Evaluation

#### Traditional ASR evaluation
```
$ source slurm_scripts/test.sh
```

#### ASR evaluation with SYN2REAL task vector
```
$ source slurm_scripts/test_vector.sh
```