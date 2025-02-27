# Clustering Context in Off-Policy Evaluation for Binary-Reward Settings

This repository contains the implementation in Python of the experimental pipeline used to run all the experiments in the paper.

**Note**: The code has been implemented and tested in an Ubuntu machine, we strongly recommend testing in a Unix-based environment (Linux/MacOS shouldn't be a problem) as we haven't adapted the code to run on a Windows environment yet and paths in the code would be problematic.

## Executing experiments

### Requirements:
1.  A Linux/Mac terminal with wget installed
2.  Python >=3.9 <10 + Poetry (https://python-poetry.org/) (creating a conda env is useful for this purpose)

### Instructions
After downloading the repository please execute:
1. `cd CHIPS`
2. `sh setup.sh`
3. `cd chips`

This will download the full version of the OBD dataset (https://research.zozo.com/data.html) for the real experiments, store it in the _/data_ directory (note that if moved, _src/Config/config.yaml_ also need to be updated), and create a poetry environment _.venv_ either in the project root folder or in the system cache. At this point, once in the _src/_ directory, any experiment can be executed using the command:

4. `poetry run python run_experiments.py -[experiment-name]`

To display all experiment names and their description one can execute the command

`poetry run python run_experiments.py -help`

For example to execute the synthetic experiment varying the distributional shift between logging and evaluation policies we can execute:

`poetry run python run_experiments.py -beta`

We can also execute more than one experiment in the same batch, for example to execute the experiments varying the distributional shift between policies, the alpha parameter for the MAP estimation process in CHIPS and the clustering generation radius we can execute :
`poetry run python run_experiments.py -beta -alpha -rad`

In our experimentation pipeline there are three main types of experiments: synthetic (using the synthetic dataset), real (using the real dataset), and multiclass (transforming classification problems into bandit feedback). It's also possible to execute all the synthetic/real/multiclass experiments using:
`poetry run python run_experiments.py -synthetic`

 `poetry run python run_experiments.py -real`
 
`poetry run python run_experiments.py -multiclass`

Finally, to execute every experiment we can execute 
`poetry run python run_experiments.py -all`


### Results storing

In the root directory of the project there is a _experiments_results/_ directory, the directory has the subdirectories _synthetic/_, _real/_, and _multiclass/_. Each of these subdirectories contains a directory per experiment associated with the father class (e.g. _synthetic/_ contains the subdirectories  _beta/_ , _rad/_, _sigma_ ...) . Every time an experiment is executed, the resulting data and associated graphs are stored in these experiment directores.

### Considerations on execution times for reproducibility
Experiments are generally executed 100 times per configuration setting, per parameter variation. This means that some experiments, like the real ones associated with the full dataset, in which supervised methods (DM, DR, DRos ...) are compared, may take considerable time to execute.

Synthetic experiments varying a single parameter take approximately 20-30 min in our machine, biparametric experiments take considerably more time.

The multiclass experiments are generally fast, except for MNIST and CIFAR100 that may take up to a couple of hours between downloading and processing.

### Troubleshooting in MacOS

For newer versions of MacOS pyyaml = "5.4.1" dependency might give some trouble since newer dependencies with Cython > 3.0 and PEP517.
We found a workaround to this at https://github.com/yaml/pyyaml/issues/736.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.