from typing import Dict, Optional, List, Tuple

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from experiment.synthetic import SyntheticExperiment
from ope.evaluation import run_multiclass_experiment


param_aka = {
    "emp_c_num": "Nº Clusters",
    "n_actions": "Nº Actions",
    "n_samples": "Nº Samples",
    "beta": r"$\beta$",
    "rad": "Cluster Ratio",
    "n_clusters_n_actions": "Nº Actions",
    "n_clusters_n_samples": "Nº Samples",
    "n_samples_alpha_bayes": r"$\alpha$ (Beta prior)",
    "sigma": r"$\sigma$",
    "alpha_bayes": r"$\alpha$ (Beta prior)",
    "clustering_method_rad": "Cluster Ratio",
    "clustering_method": "Clustering Method",
    "cluster_rad": "Cluster Ratio",
    "beta_alpha": "Beta",
}

palette = {
    "IPS": "tab:orange",
    "CHIPS_bayes": "tab:red",
    "DM": "tab:blue",
    "DR": "tab:green",
    "MIPS (w/o SLOPE)": "tab:cyan",
    "MIPS (w/ SLOPE)": "tab:pink",
    "MR": "#B9D2EE",
    "SwitchDR": "tab:brown",
    "DRos": "#F5DA33",
}


class MultiClassExperiment(SyntheticExperiment):
    def __init__(
        self, 
        base_config : Dict, 
        name : str, 
        params : Dict, 
        baseline : Optional[bool] = False, 
        ref : Optional[str] = None, 
        base_dir : Optional[str] = None, 
        **kwargs
    ) -> None:
        """Initializes the MultiClassExperiment class

        Parameters
        ----------
        base_config : Dict[str, Any]
            A dictionary containing the base configuration for the experiment (loaded from the config file).
        name : str
           The experiment's name.
        params : Dict[str, List]
            A dictionary containing the names and values parameter(s) to be varied in the experiment.
        baseline : bool, optional
            Normalize the results with respect to IPS, by default False.
        ref : str, optional
            The reference parameter (the one in the x axis in the plots). 
            Must be provided in multiparameter experiments by default None
        base_dir : str, optional
            Path to the directory to store the results, by default None.
        """

        self.ests = []
        self.baseline = baseline
        super().__init__(base_config, name, params, None, ref, base_dir=base_dir, **kwargs)

    def process_run_outputs(self, outs: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
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
        return {
            "mse": pd.DataFrame(outs[0])[
                ["IPS", "DR", "DM", "CHIPS_bayes", "MR", "SwitchDR", "DRos"]
            ]
        }

    def run_iter(self):
        """
        Runs an iteration of the experiment with the configuration stored in self.config
        """
        return run_multiclass_experiment(**self.config)

    def plot_result(
            self, 
            stat : str, 
            result : pd.DataFrame, 
            **kwargs
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plots the results of the experiments

        Parameters
        ----------
        stat : str
            The statistic to be plotted.
        result : pd.DataFrame
            The results of the experiment in a dataframe form.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Plot and axes of the resulting figure.
        """
        fig, ax = plt.subplots(figsize=(15, 8), tight_layout=True)
        # ECDF plot
        sns.barplot(data=result, ax=ax, palette=palette)
        ax.set_yscale("log")
        ax.set_title(f"Normalized MSE {self.name}")
        return fig, ax
