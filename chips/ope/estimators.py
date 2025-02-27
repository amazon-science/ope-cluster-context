from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cluster, mixture
from ope.legacy import MarginalizedRatioWeighting
from obp.ope import SelfNormalizedInverseProbabilityWeighting as SNIPW
from obp.ope import SelfNormalizedDoublyRobust as SNDR_
from obp.ope import DoublyRobustWithShrinkage as DRos_
from ope.legacy import MIPS as MIPS_

predicted_rewards_obp = None

@dataclass
class BanditsConfig:
    X: np.ndarray
    x_b_idx: np.ndarray
    A_b: np.ndarray
    R_b: np.ndarray
    pi_e: np.ndarray
    pi_b: np.ndarray
    a_num: int
    n_samples: int
    emp_c_num: int
    reward_models: list
    # For efficiency computation of DR and CHIPS_mean, running full is not worth it
    # since part of the logic is computed in DM or CHIPS_bayes
    dm_reward: float = 0
    chips_bayes: float = 0
    chips_mean: float = 0
    alpha_bayes: float = 20
    beta_bayes: float = 20
    clustering_method: str = "kmeans"
    def_actions: int = 0
    predicted_rewards_obp = None

    X_b: np.ndarray = field(init=False)

    def __post_init__(self):
        self.X_b = self.X[self.x_b_idx]


class Estimator(ABC):
    def __init__(self, name) -> None:
        self.name = name

        @abstractmethod
        def run_ope(self, config):
            pass


class DM(Estimator):
    """
    Direct Method implementation.
    """
    def __init__(self) -> None:
        super().__init__(name="DM")

    def run_ope(self, config : BanditsConfig):
        acc_value = 0
        for a in np.unique(config.A_b):
            acc_value += np.sum(
                config.reward_models[a].predict(config.X_b) * config.pi_e[config.x_b_idx, a]
            )
        config.dm_reward = acc_value
        return acc_value / config.n_samples


class IPS(Estimator):
    """
    Inverse Propensity Score implementation.
    """
    def __init__(self) -> None:
        super().__init__(name="IPS")

    def run_ope(self, config : BanditsConfig):
        acc_value = 0
        acc_value = (config.R_b * config.pi_e[config.x_b_idx, config.A_b]) @ (
            1 / config.pi_b[config.x_b_idx, config.A_b]
        )
        # print('IPS:', acc_value / config.n_samples)
        return acc_value / config.n_samples


class DR(Estimator):
    """
    Doubly Robust implementation.
    """
    def __init__(self) -> None:
        super().__init__(name="DR")

    def run_ope(self, config : BanditsConfig):
        acc_value = config.dm_reward
        for a in np.unique(config.A_b):
            mask = config.A_b == a
            if config.dm_reward == 0:
                acc_value += np.sum(
                    config.reward_models[a].predict(config.X_b) * config.pi_e[config.x_b_idx, a]
                )
            r_ = config.R_b[mask] - config.reward_models[a].predict(config.X_b[mask])
            acc_value += np.sum(
                r_ * config.pi_e[config.x_b_idx[mask], a] / config.pi_b[config.x_b_idx[mask], a]
            )
        # print('DR:', acc_value / config.n_samples)
        return acc_value / config.n_samples


class CHIPS(Estimator):
    """
    CHIPS implementation
    """
    def __init__(
        self, 
        use_bayes : Optional[bool] = True, 
        alpha : Optional[float] = 20.0, 
        beta : Optional[float] = 20.0, 
        debug : Optional[bool] = False
        ) -> None:
        """Initializes the CHIPS estimator hyperparameters.

        Parameters
        ----------
        use_bayes : Optional[bool], optional
            Use MAP estimation for cluster reward estimation, by default True, if False estimate with ML.
        alpha : Optional[float], optional
            Alpha parameter of the beta prior used in MAP method, by default 20.0 .
        beta : Optional[float], optional
            Beta parameter of the beta prior used in MAP method, by default 20.0 .
        debug : Optional[bool], optional
            Debug mode, it returns the cluster reward estimates besides the policy value estimate, by default False.
        """
        super().__init__(
            name=f'CHIPS_{"bayes" if use_bayes else "mean"}',
        )
        self.use_bayes = use_bayes
        self.alpha, self.beta = alpha, beta
        self.debug = debug

    def __bayes_reward(self, alpha : float, beta : float, y : int, N : int) -> float:
        """Estimate the reward using a Beta prior.

        Parameters
        ----------
        alpha : float
            Alpha parameter of the beta prior.
        beta : float
            Beta parameter of the beta
        y : int
            Number of rewards that are 1 for a cluster.
        N : int
            Number of samples in for a cluster.
            
        Returns
        -------
        float
           Estimated reward for a cluster.
        """
        new_alpha = alpha + y
        new_beta = beta + N - y
        return (new_alpha - 1) / (new_beta + new_alpha - 2)

    def __estimate_cluster_probs(self, 
            probs : np.ndarray, 
            n_clusters : int, 
            cluster_idxs : np.ndarray, 
            correction : Optional[bool] = True
        ) -> np.ndarray:
        """Estimate the probabilities of a context belonging to a cluster.

        Parameters
        ----------
        probs : np.ndarray
            p(a_i|x_i) probabilities for the logging samples (dimension n_samples x n_actions).
        n_clusters : int
            Number of clusters to use.
        cluster_idxs : np.ndarray
            The cluster inidices for each context in the samples (dimension 1 x n_samples).
        correction : Optional[bool], optional
            Substitutes 0 probability in the estimates by 2e^-16 for numerical stability, by default True

        Returns
        -------
        np.ndarray
            Estimated probabilities of a context belonging to a cluster (dimension n_clusters x n_actions).
        """
        estimates = np.zeros((n_clusters, probs.shape[1]))
        u_c = np.unique(cluster_idxs)
        for i in u_c[u_c > -1].astype(int):
            estimates[i] = (p := probs[cluster_idxs == i].sum(axis=0)) / sum(p)
        if correction:
            estimates[estimates == 0] = 2e-16
        return estimates

    def run_ope(self, config : BanditsConfig):
        # use uninformative prior
        self.alpha = config.alpha_bayes
        self.beta = config.alpha_bayes

        if config.clustering_method == "affinity_propagation":
            clustering_method = cluster.AffinityPropagation()
        elif config.clustering_method == "mean_shift":
            clustering_method = cluster.MeanShift()
        elif config.clustering_method == "spectral_clustering":
            clustering_method = cluster.SpectralClustering(n_clusters=config.emp_c_num)
        elif config.clustering_method == "agglomerative_clustering":
            clustering_method = cluster.AgglomerativeClustering(n_clusters=config.emp_c_num)
        elif config.clustering_method == "dbscan":
            clustering_method = cluster.DBSCAN()
        elif config.clustering_method == "optics":
            clustering_method = cluster.OPTICS()
        elif config.clustering_method == "birch":
            clustering_method = cluster.Birch(n_clusters=config.emp_c_num)
        elif config.clustering_method == "bayes_gmm":
            clustering_method = mixture.BayesianGaussianMixture(n_components=config.emp_c_num)
        elif config.clustering_method == "gmm":
            clustering_method = mixture.GaussianMixture(n_components=config.emp_c_num)
        else:
            clustering_method = cluster.MiniBatchKMeans(n_clusters=config.emp_c_num)

        # cluster contexts
        c_x_ = clustering_method.fit_predict(config.X)
        # estimate the probabilities p(a|c) for logging
        if config.clustering_method in ["affinity_propagation", "mean_shift", "dbscan", "optics"]:
            emp_c_num = np.unique(c_x_.squeeze()).shape[0]
        else:
            emp_c_num = config.emp_c_num
        p_a_c_b = self.__estimate_cluster_probs(config.pi_b, emp_c_num, c_x_)
        p_a_c_e = self.__estimate_cluster_probs(config.pi_e, emp_c_num, c_x_)

        r_c = np.zeros(emp_c_num)
        r_c_mean = np.zeros(emp_c_num)

        for c in np.unique(c_x_[config.x_b_idx]):
            mask = c_x_[config.x_b_idx] == c
            y = (config.R_b[mask] == 1).sum()
            N = mask.sum()
            r_c_mean[c] = config.R_b[mask].mean()
            if self.use_bayes:
                r_c[c] = self.__bayes_reward(self.alpha, self.beta, y, N)

        c_b = c_x_[config.x_b_idx]
        w = p_a_c_e[c_b, config.A_b] / p_a_c_b[c_b, config.A_b]
        config.chips_bayes = (r_c[c_b] @ w) / config.n_samples
        config.chips_mean = (r_c_mean[c_b] @ w) / config.n_samples

        if self.debug and self.use_bayes:
            return w, r_c[c_b]
        elif self.debug:
            return w, r_c_mean[c_b]
        else:
            if self.use_bayes:
                return config.chips_bayes
            else:
                # print(w, r_c_mean, config.chips_mean)
                return config.chips_mean

# Adapted estimators from the OPE pipeline
class MR(Estimator):
    """
    Marginal Density Ratio implementation.
    """
    def __init__(self) -> None:
        super().__init__(name="MR")
    
    def run_ope(self, config):
        return MarginalizedRatioWeighting(
            reward = config.R_b,
            pi_b = config.pi_b[config.x_b_idx, config.A_b],
            pi_e = config.pi_e[config.x_b_idx, config.A_b],
            reward_type="binary",
            #use_estimated_pscore=True
        ).estimate_policy_value(config.R_b, config.A_b, config.pi_b[:,:,np.newaxis], pscore=config.pi_b[config.x_b_idx, config.A_b])
    

class SNIPS(Estimator):
    """
    Self-Normalized Inverse Propensity Score implementation.
    """
    def __init__(self) -> None:
        super().__init__(name="SNIPS")

    def run_ope(self, config):
        return SNIPW().estimate_policy_value(
            reward=config.R_b, 
            action=config.A_b, 
            action_dist=config.pi_b[config.x_b_idx,:,np.newaxis], 
            pscore=config.pi_b[config.x_b_idx, config.A_b]
            )
    
class SNDR(Estimator):
    """
    Self-Normalized Doubly Robust implementation
    """
    def __init__(self) -> None:
        super().__init__(name="SNDR")

    def get_predicted_rewards(self, A_b, X_b, config):
        rewards = np.zeros((config.n_samples, config.a_num, 1))
        if type(config.predicted_rewards_obp) != np.ndarray:
            for a in np.unique(A_b):
                rewards[:, a, 0] = config.reward_models[a].predict(X_b)
            config.predicted_rewards_obp = rewards
        return config.predicted_rewards_obp
    
    def run_ope(self, config):
        lambdas = [10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]
        return SNDR_(use_estimated_pscore=True, lambda_=100000).estimate_policy_value(
            reward=config.R_b, 
            action=config.A_b, 
            action_dist=config.pi_b[config.x_b_idx,:,np.newaxis],
            estimated_rewards_by_reg_model=self.get_predicted_rewards(config.A_b, config.X_b, config),
            estimated_pscore=config.pi_e[config.x_b_idx, config.A_b]
            )
    
class DRoS(Estimator):
    """
    Doubly Robust with Shrinkage implementation
    """
    def __init__(self) -> None:
        super().__init__(name="DRoS")

    def get_predicted_rewards(self, A_b, X_b, config):
        rewards = np.zeros((config.n_samples, config.a_num, 1))
        if type(config.predicted_rewards_obp) != np.ndarray:
            for a in np.unique(A_b):
                rewards[:, a, 0] = config.reward_models[a].predict(X_b)
            config.predicted_rewards_obp = rewards
        return config.predicted_rewards_obp

    def run_ope(self, config):
        lambdas = [10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]
        return DRos_(lambda_=100000, use_estimated_pscore=True).estimate_policy_value(
            reward=config.R_b, 
            action=config.A_b, 
            action_dist=config.pi_b[config.x_b_idx,:,np.newaxis],
            estimated_rewards_by_reg_model=self.get_predicted_rewards(config.A_b, config.X_b, config),
            estimated_pscore=config.pi_e[config.x_b_idx, config.A_b]
        )