#!/usr/bin/env python
import argparse
import warnings

import yaml

from experiment.synthetic import SyntheticExperiment
from experiment.real import RealExperiment
from experiment.multiclass import MultiClassExperiment
from experiment import ExperimentBatch
from ope.generation import generate_real


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(
    description="Run all the experiments associated with the CHIPS estimator"
)

parser.add_argument(
    "-clusters", action="store_true", help="execute varying number of cluster experiment."
)
parser.add_argument(
    "-actions", action="store_true", help="execute varying number of actions experiment."
)
parser.add_argument("-rad", action="store_true", help="execute varying radius experiment.")
parser.add_argument(
    "-beta", action="store_true", help="execute different values of beta experiment."
)
parser.add_argument(
    "-samples", action="store_true", help="execute varying number of samples experiment."
)
parser.add_argument(
    "-sigma", action="store_true", help="execute different values of sigma experiment."
)
parser.add_argument(
    "-alpha", action="store_true", help="execute different values of bayes beta alpha experiment."
)
parser.add_argument(
    "-clusters_actions",
    action="store_true",
    help="execute varying number of clusters and actions experiments (warning: heavy computation)",
)
parser.add_argument(
    "-samples_alpha",
    action="store_true",
    help="execute varying number of samples and the alpha parameters experiments (warning: heavy computation)",
)
parser.add_argument(
    "-clustering_method_rad",
    action="store_true",
    help="execute varying radii and clustering methods experiment (warning: heavy computation)",
)
parser.add_argument(
    "-clusters_samples",
    action="store_true",
    help="execute varying number of clusters and actions experiments (warning: heavy computation)",
)
parser.add_argument(
    "-beta_alpha",
    action="store_true",
    help="execute varying beta (policy shift) and alpha (beta prior for CHIPS) (warning: heavy computation)",
)
parser.add_argument(
    "-def_actions",
    action="store_true",
    help="execute varying number of deficient actions experiment.",
)
parser.add_argument(
    "-real_4",
    action="store_true",
    help="execute real data experiment with 4 clusters and 50k samples (warning: heavy computation)",
)
parser.add_argument(
    "-real_8",
    action="store_true",
    help="execute real data experiment with 8 clusters and 100k samples (warning: heavy computation)",
)
parser.add_argument(
    "-real_40",
    action="store_true",
    help="execute real data experiment with 40 clusters and 500k samples (warning: heavy computation)",
)
parser.add_argument(
    "-real_a_10",
    action="store_true",
    help="execute real data experiment with 8 clusters, 50k samples and 10 alpha (warning: heavy computation)",
)
parser.add_argument(
    "-real_a_100",
    action="store_true",
    help="execute real data experiment with 8 clusters, 50k samples and 100 alpha (warning: heavy computation)",
)
parser.add_argument(
    "-real_multi",
    action="store_true",
    help="execute real data experiment with multiple clusters and 100k samples (warning: heavy computation)",
)
parser.add_argument(
    "-real_alpha",
    action="store_true",
    help="execute real data experiment with multiple alpha values and 100k samples (warning: heavy computation)",
)
parser.add_argument(
    "-real_non_parametric",
    action="store_true",
    help="execute real data experiment using OPTICS as clustering method and 100k samples (warning: heavy computation)",
)
parser.add_argument("-real", action="store_true", help="execute all real experiments")

parser.add_argument("-digits", action="store_true", help="execute multilabel digits experiment.")
parser.add_argument("-letter", action="store_true", help="execute multilabel letters experiment.")
parser.add_argument("-mnist", action="store_true", help="execute multilabel mnist experiment.")
parser.add_argument(
    "-cifar100", action="store_true", help="execute multilabel cifar100 experiment."
)
parser.add_argument(
    "-optdigits", action="store_true", help="execute multilabel optdigits experiment."
)
parser.add_argument(
    "-pendigits", action="store_true", help="execute multilabel pendigits experiment."
)
parser.add_argument(
    "-satimage", action="store_true", help="execute multilabel satimage experiment."
)

parser.add_argument("-synthetic", action="store_true", help="execute all synthetic experiments")
parser.add_argument("-multiclass", action="store_true", help="execute all multilabel experiments")
parser.add_argument("-all", action="store_true", help="execute all the experiments")


def main():
    args = vars(parser.parse_args())
    batch = []
    with open("./config/settings.yaml") as f:
        settings = yaml.safe_load(f.read())

    synthetic_d = settings["synthetic"]
    real_d = settings["real"]
    multiclass_d = settings["multilabel"]

    if args["all"] or not any(args.values()):
        target = list(synthetic_d.keys()) + list(real_d.keys())
    elif args["synthetic"]:
        target = synthetic_d.keys()
    elif args["real"]:
        target = real_d.keys()
    elif args["multiclass"]:
        target = multiclass_d.keys()
    else:
        target = []

    for k in target:
        args[k] = True

    synthetic_experiments = [SyntheticExperiment(**v) for k, v in synthetic_d.items() if args[k]]
    real_experiments = []
    for k, v in real_d.items():
        if args[k]:
            real_data = generate_real(v["base_config"])
            real_experiments.append(RealExperiment(**{**v, **real_data}))
    multilabel_experiments = [MultiClassExperiment(**v) for k, v in multiclass_d.items() if args[k]]
    batch = ExperimentBatch(*synthetic_experiments, *real_experiments, *multilabel_experiments)
    batch.run()


if __name__ == "__main__":
    main()
