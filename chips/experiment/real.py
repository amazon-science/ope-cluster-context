from typing import Dict, Optional, List, Any

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import obp

from experiment import Experiment
from ope.evaluation import run_real_experiment

param_aka = {
    "emp_c_num": "Nº Clusters",
    "n_actions": "Nº Actions",
    "n_samples": "Nº Samples",
    "beta": r"$\beta$",
    "rad": "Cluster Ratio",
    "n_clusters_n_actions": "Nº Actions",
    "n_clusters_n_samples": "Nº Samples",
    "sigma": r"$\sigma$",
    "alpha_bayes": r"$\alpha, \beta$ (Beta prior)",
    "clustering_method": "Clustering Method",
}


class RealExperiment(Experiment):
    def __init__(
        self,
        base_config : Dict[str, Any],
        name : str,
        params : Dict[str, List],
        policy_ur : obp.policy.BaseContextualPolicy,
        policy_ts : obp.policy.BaseContextualPolicy,
        V_e : np.ndarray,
        dataset_b : np.ndarray,
        dataset_e : np.ndarray,
        x_b_idx_ : np.ndarray,
        X : np.ndarray,
        estimators : List[str],
        base_dir : Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes the RealExperiment class

        Parameters
        ----------
        base_config : Dict[str, Any]
            A dictionary containing the base configuration for the experiment (loaded from the config file).
        name : str
            The experiment's name.
        params : Dict[str, List]
            A dictionary containing the names and values parameter(s) to be varied in the experiment.
        policy_ur : obp.policy.BaseContextualPolicy
            The logging policy.
        policy_ts : obp.policy.BaseContextualPolicy
            The evaluation policy.
        V_e : np.ndarray
            The evaluation policy value.
        dataset_b : np.ndarray
            The logging datasets
        dataset_e : np.ndarray
            The evaluation dataset
        x_b_idx_ : np.ndarray
            The indices associated with the logging dataset (useful for indexing when estimating the policies).
        X : np.ndarray
            The features (context) from the logging dataset.
        estimators : List[str]
            A list of estimators to be used in the experiment.
        base_dir : Optional[str], optional
            Path to the directory to store the results, by default None.
        """
        super().__init__(
            base_config, name, params, estimators=estimators, base_dir=base_dir, plot_kwargs=kwargs
        )
        # configurations
        self.policy_ur = policy_ur
        self.policy_ts = policy_ts
        self.V_e = V_e
        self.dataset_b = dataset_b
        self.dataset_e = dataset_e
        self.x_b_idx_ = x_b_idx_
        self.X = X
        self.estimators = estimators

    def process_run_outputs(self, outs):
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
        if not self.estimators:
            # base case for running reaal experiments
            return {"mse": pd.DataFrame(outs[0])}
        else:
            # running real experiments varying some parameters
            res_dict = {"IPS": np.ones(self.config["runs"])}
            for run_data, param_value in zip(
                outs, self.config_iters["values"][:: self.config["runs"]]
            ):
                for est, vals in run_data.items():
                    ips_coef = 1 / np.array(run_data["IPS"])
                    if est != "IPS":
                        res_dict[f'{param_aka[self.config_iters["names"][0]]}_{param_value[0]}'] = (
                            np.array(vals) * ips_coef
                        )
            res = {
                "mse": pd.DataFrame.from_dict(res_dict),
            }
            return res

    def run_iter(self) -> Dict[str, pd.DataFrame]:
        """
        Runs an iteration of the experiment with the configuration stored in self.config
        """
        res_estimators = run_real_experiment(
            sample_size=self.config["sample_size"],
            n_clusters=self.config["emp_c_num"],
            n_seeds=self.config["runs"],
            n_actions=self.config["n_actions"],
            X=self.X,
            V_e=self.V_e,
            dataset_b=self.dataset_b,
            policy_ts=self.policy_ts,
            policy_ur=self.policy_ur,
            x_b_idx_=self.x_b_idx_,
            estimators=self.estimators,
            bayes_alpha=self.config["alpha_bayes"],
            clustering_method=self.config["clustering_method"],
        )

        return res_estimators

    def plot_result(
            self, 
            stat : str, 
            result : pd.DataFrame, 
            **kwargs
        ):
        """Plots the results of the experiments

        Parameters
        ----------
        stat : str
            The statistic to be plotted.
        result : pandas.DataFrame
            The results of the experiment in a dataframe form.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Plot and axes of the resulting figure.
        """
        df = result
        norm_df = df.apply(lambda c: c / df.IPS, axis=0)
        fig, ax = plt.subplots(figsize=(15, 8), tight_layout=True)
        # ECDF plot
        sns.ecdfplot(
            linewidth=2,
            data=norm_df,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_title("Relative ratio probability distirbution")
        ax.set_ylabel(r"$P(\frac{MSE(V)_{CLIPS}}{MSE(V)_{IPS}} < x)$")
        return fig, ax
