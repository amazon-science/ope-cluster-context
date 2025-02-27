#!/bin/bash

# Download the Open Bandits Dataset in data dir
wget https://research.zozo.com/data_release/open_bandit_dataset.zip -P ./data/.

# Unzip and remove zip
unzip ./data/open_bandit_dataset.zip -d data/.
rm ./data/open_bandit_dataset.zip

# cifar100 dataset

# create poetry env
poetry install

