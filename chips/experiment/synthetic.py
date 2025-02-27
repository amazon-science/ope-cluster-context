from typing import List, Dict, Optional, Any, Tuple

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns



from experiment import Experiment
from ope.evaluation import run_synthetic_experiment


param_aka = {
    'emp_c_num' : 'Nº Clusters',
    'n_actions' : 'Nº Actions',
    'n_samples' : 'Nº Samples',
    'beta' : r'$\beta$',
    'rad' : 'Cluster Ratio',
    'n_clusters_n_actions' : 'Nº Actions',
    'n_clusters_n_samples' : 'Nº Samples',
    'n_samples_alpha_bayes' : r'$\alpha$ (Beta prior)',
    'sigma' : r'$\sigma$',
    'alpha_bayes' : r'$\alpha$ (Beta prior)',
    'clustering_method_rad' : 'Cluster Ratio',
    'clustering_method' : 'Clustering Method',
    'cluster_rad' : 'Cluster Ratio',
    'beta_alpha' : 'Beta',
    'def_actions' : 'Number of Deficient Actions',

}


class SyntheticExperiment(Experiment):
    def __init__(
            self, 
            base_config : Dict[str, Any],
            name, params : Dict[str, List],
            estimators : List[str], 
            baseline : Optional[bool] = False, 
            ref : Optional[str] = None, 
            base_dir : Optional[str] = None, 
            **kwargs
    ) -> None:
        """
        Initialize the SyntheticExperiment class.

        Parameters
        ----------
        base_config : Dict[str, Any]
            Base configuration for the experiment.
        name : str
            Name of the experiment.
        params : Dict[str, List]
            Parameters for the experiment.
        estimators : List[str]
            List of estimators to be used in the experiment.
        baseline : Optional[bool], optional
            Whether to use baseline comparison, by default False.
        ref : Optional[str], optional
            Reference parameter for plotting, by default None.
        base_dir : Optional[str], optional
            Base directory for saving results, by default None.
        """

        self.ests = estimators
        self.baseline = baseline
        super().__init__(base_config, name, params, estimators, ref, base_dir, **kwargs)
    
    def process_run_outputs(self, outs):

        stats = [pd.concat(x).reset_index(drop=True) for x in zip(*outs)]
        param_values = pd.DataFrame(
            np.array(self.config_iters['values']).squeeze(), 
            columns=self.config_iters['names']
        )
        unique_param_vals = param_values.iloc[::self.config['runs']].reset_index(drop=True)
        res = {
            'mse' : pd.concat([stats[0], param_values], axis=1),
            'sq_bias' : pd.concat([stats[1], unique_param_vals], axis=1),
            'var' : pd.concat([stats[2], unique_param_vals], axis=1)
        }
        return res
    
    def run_iter(self):
        mse, bias, var = run_synthetic_experiment(**self.config)
        ret = (
            pd.DataFrame.from_dict(mse)[self.estimators],
            pd.DataFrame.from_dict(bias, orient='index').T[self.estimators],
            pd.DataFrame.from_dict(var, orient='index').T[self.estimators],
        )
        if self.baseline:
            for i in range(len(ret)):
                result = ret[i]
                base = result.IPS.copy()
                for est in self.ests:
                    result[est] = result[est] / base
        return ret

    def __convert_2_seaborn_format(self, results:pd.DataFrame) -> pd.DataFrame:
        """
        Convert the results to a format that plotted by seaborn directly as band plots.

        Parameters
        ----------
        results : pandas.DataFrame
            The results of the experiment.

        Returns
        -------
        pandas.DataFrame
            Formatted pandas.DataFrame
        """
        est = results.columns[:-len(self.config_iters['names'])]
        a = pd.concat([results[e] for e in est], axis=0).reset_index(drop=True)
        b = pd.concat([pd.Series([e]*results.shape[0]) for e in est]).reset_index(drop=True)
        c = pd.concat([results[self.config_iters['names']] for _ in est]).reset_index(drop=True)
        df = pd.concat([a, b, c], axis=1)
        df.columns = ['y', 'est'] + self.config_iters['names']
        return df

    def plot_result(
            self, 
            stat : str, 
            result : pd.DataFrame,
            logx : Optional[bool] = False,
            logy : Optional[bool] = False,
            figsize : Optional[tuple] = (15, 8),
            fig : Optional[matplotlib.figure.Figure] = None,
            ax : Optional[matplotlib.axes.Axes] = None,
            **kwargs       
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """_summary_

        Parameters
        ----------
        stat : str
            Main statistic to be plotted (e.g. MSE) will be used for the name of the resulting file and the title.
        result : pandas.DataFrame
            The results of the experiment.
        logx : Optional[bool], optional
            Apply logarithmic scaling in the x axis, by default False
        logy : Optional[bool], optional
            Apply logarithmic scaling in the y axis, by default False
        figsize : Optional[tuple], optional
            Size of the figure to generate, by default (15, 8)
        fig : Optional[matplotlib.figure.Figure], optional
            Figure in which to plot the results, by default None and new figure will be created.
        ax : Optional[matplotlib.axes.Axes], optional
            Axes in which to plot the results, by default None and new axes will be created.
        
        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Figure and axes of the resulting
        """
        exp_name = self.name
        plt.rcParams['xtick.bottom'] = True
        if fig == None or ax == None:
            fig, ax = plt.subplots(figsize=figsize)
        base = result.IPS.copy()
        cols = self.ests + self.config_iters['names']
        if self.baseline and len(self.params) > 1:
            cols = ~(result.columns == 'IPS')

        df_ = self.__convert_2_seaborn_format(result.loc[:, cols])
            
        extra_params = [c for c in df_.columns[~(df_.columns.isin(('y', 'est', self.ref)))]]
        hue = ['est'] + extra_params
        sns.lineplot(
            x=self.ref, 
            y='y', 
            data=df_, 
            dashes=False, 
            ax=ax, 
            hue=df_[hue].apply(
                lambda r: f'{r[0]} {",".join(param_aka[st] + ":" + str(val) for st, val in zip(self.config_iters["names"], r[1:]))}', 
                axis=1)
        )
        if logx : 
            ax.set_xscale('log')
        if logy : 
            ax.set_yscale('log')
        ax.set_xlabel(f'{param_aka[self.name]}')
        ax.set_title(f' {param_aka[exp_name]} {stat.capitalize()}')
        ax.set_xticks(np.round(self.params[self.ref], 3))
        ax.set_xticklabels(np.round(self.params[self.ref], 3))
        return fig, ax
    
