from abc import ABC, abstractmethod
from itertools import product
from math import prod
from typing import Any, Dict, List, Tuple, Optional, Iterable


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm



sns.set_style("darkgrid")


class Experiment(ABC):
    """ 
    Class for managing the experiments
    """
    def __init__(
        self, 
        base_config : Dict[str, Any],
        name : str, 
        params : Dict[str, List],
        estimators : List[str], 
        ref : Optional[str] = None, 
        base_dir : Optional[str] = None, 
        **kwargs
    ) -> None:
        """
        Initializes the Experiment class

        Parameters
        ----------
        base_config : Dict[str, Any]
            A dictionary containing the base configuration for the experiment (loaded from the config file).
        name : str
           The experiment's name.
        params : Dict[str, List]
            A dictionary containing the names and values parameter(s) to be varied in the experiment.
        estimators : List[str]
            A list of estimators to be used in the experiment.
        ref : str, optional
            The reference parameter (the one in the x axis in the plots). 
            Must be provided in multiparameter experiments by default None
        base_dir : str, optional
            Path to the directory to store the results, by default None.

        Raises
        ------
        ValueError
            If a multiparameter experiment is being run and the reference parameter is not provided.
        """
        self.config = base_config.copy()
        self.name = name
        self.plot_kwargs = kwargs
        if len(params.keys()) == 1:
            self.ref = list(params.keys())[0]
        elif ref != None:
            self.ref = ref
        else:
            raise ValueError(
                "Reference value (ref) must be provided in multi parameter experiments."
            )

        self.params = params
        self.base_dir = f"./{base_dir}/{name}" if base_dir != None else f"./{name}"
        self.config_iters = {
            "params": [],
            "values": [],
        }
        self.result = {}
        self.estimators = estimators

    @abstractmethod
    def process_run_outputs(self, outs : List) -> Dict[str, pd.DataFrame]:
        """
        Processes the outputs of the experiment

        Parameters
        ----------
        outs : List
            The outputs (policy value estimation of the experiments).

        Returns
        -------
        Dict[str, pd.DataFrame]
            Processed outputs for bias, variance and mse in dataframe format.
        """
        pass

    @abstractmethod
    def run_iter(self) -> float:
        """
        Runs an iteration of the experiment with the configuration stored in self.config
        """
        pass

    @abstractmethod
    def plot_result(self, stat, result, **plot_kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the results of the experiments

        Parameters
        ----------
        stat : str
            The statistic to be plotted.
        result : pd.DataFrame
            The results of the experiment in a dataframe form.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Plot and axes of the resulting figure.
        """
        pass

    def plot_results(self, save=True):
        """Plots MSE, Bias and Variance results of the experiment

        Parameters
        ----------
        save : bool, optional
            Flag to save the plotted images in pdf format, by default True
        """
        for stat, val in self.result.items():
            fig, _ = self.plot_result(stat, val, **self.plot_kwargs)
            fig.savefig(f"{self.base_dir}/{stat}.pdf", format="pdf")

    def run(self, save=True):
        """
        Runs an experiment varying the target parameter(s) specified in the params dictionary and saves the
        raw results.

        Parameters
        ----------
        save : bool, optional
            Flag to save the results of the experiment in csv format, by default True
        """
        self.config_iters["names"] = list(self.params.keys())
        runs = self.config["runs"]
        param_values = self.params.values()
        outs = [None] * prod(len(v) for v in param_values)
        self.config_iters["values"] = [None] * len(outs) * runs
        for i, param_vals in tqdm(enumerate(product(*param_values)), total=len(outs)):
            for param, value in zip(self.config_iters["names"], param_vals):
                self.config[param] = value
            outs[i] = self.run_iter()
            self.config_iters["values"][i * runs : (i + 1) * runs] = [param_vals] * runs

        self.result = self.process_run_outputs(outs)

        for stat in self.result:
            if save:
                self.result[stat].to_csv(f"{self.base_dir}/{stat}.csv", index=False)


class ExperimentBatch:
    """
    A wrapper class to create experiment batches.
    """
    def __init__(self, *experiments:Iterable[Experiment]) -> None:
        """
        Initializes the ExperimentBatch class with the provided experiments.
        """
        self.experiments = experiments

    def join(self, experiment_batch):
        """
        Joins two experiment batches.

        Parameters
        ----------
        experiment_batch : ExperimentBatch
            Another experiment batch to join with.

        Returns
        -------
        ExperimentBatch
            A new ExperimentBatch with the experiments from both batches.
        """
        self.experiments += experiment_batch
        return self

    def run(self, plot=True):
        """
        Runs all experiments in the batch and plots the results if plot is set to True.
        Parameters
        ----------
        plot : bool, optional
            A flag to plot experiment results if active, by default True.
        """
        for e in self.experiments:
            e.run()
            if plot:
                e.plot_results()
