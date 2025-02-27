from ast import Dict
from dataclasses import dataclass
from obp.dataset import OpenBanditDataset
from obp.dataset.base import BaseBanditDataset
from obp.ope import BaseOffPolicyEstimator
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import DoublyRobustWithShrinkageTuning as DRos
from obp.ope import InverseProbabilityWeighting
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import InverseProbabilityWeighting as IPW
from obp.ope import OffPolicyEvaluation
from obp.ope import SubGaussianDoublyRobustTuning as SGDR
from obp.ope import SwitchDoublyRobustTuning as SwitchDR
from obp.types import BanditFeedback
from obp.utils import check_array
from obp.utils import sample_action_fast
from obp.utils import sample_action_fast, check_array
from pandas import DataFrame
from random import sample
from scipy import stats
from scipy.stats import rankdata
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from sklearn.utils import check_X_y
from typing import Dict
from typing import Optional
from typing import Union
import itertools
import logging
import numpy as np
import pandas as pd


@dataclass
class MarginalizedRatioWeighting(InverseProbabilityWeighting):
    """
    All credit for this code go to Yuta Saito and Thorsten Joachims.
    Original repository distirbuted under MIT license can be found
    in https://github.com/usaito/icml2022-mips
    """

    def __init__(
        self,
        reward,
        pi_b,
        pi_e,
        reward_type,
        estimation_method="default",
        use_estimated_pscore=False,
        estimator_name="mr",
        **kwargs,
    ):
        self.estimator_name = estimator_name
        self.estimation_method = estimation_method
        self.reward_type = reward_type
        self.use_estimated_pscore = use_estimated_pscore
        if reward_type == "binary":
            self._fit_w_y_binary(reward, pi_b, pi_e)
        elif estimation_method == "default":
            self._fit_w_y(reward, pi_b, pi_e)
        else:
            self._fit_w_y_alternative(reward, pi_b, pi_e)

    def _fit_w_y(self, reward, pi_b, pi_e):
        self.w_y_nn = MLPRegressor(
            hidden_layer_sizes=(512, 256, 32),
            verbose=True,
            alpha=0.6,
            max_iter=1000,
            early_stopping=False,
        )
        policy_ratios = pi_e / pi_b
        self.policy_ratios = policy_ratios.reshape(-1)
        self.reward = reward.reshape(-1)
        logging.info(f"Range of policy ratios: {policy_ratios.min()}-{policy_ratios.max()}")
        policy_ratios_normalised = (policy_ratios - policy_ratios.mean()) / max(
            policy_ratios.std(), 1
        )
        reward_normalised = (reward - reward.mean()) / reward.std()
        self.w_y_nn.fit(
            X=reward_normalised.reshape(-1, 1),
            y=policy_ratios_normalised.reshape(-1, 1),
        )
        self.w_y = (
            lambda x: self.w_y_nn.predict(
                (x.reshape(-1, 1) - reward.mean()) / reward.std()
            ).reshape(-1)
            * max(policy_ratios.std(), 1)
            + policy_ratios.mean()
        )
        self.w_y_weights = self.w_y(reward).reshape(-1)
        logging.info(f"Range of ratios w(y): {self.w_y_weights.min()}-{self.w_y_weights.max()}")

    def _fit_w_y_alternative(self, reward, pi_b, pi_e):
        self.w_y_nn = MLPRegressor(
            hidden_layer_sizes=(512, 256, 32),
            verbose=True,
            alpha=0.6,
            max_iter=1000,
            early_stopping=False,
        )
        policy_ratios = pi_e / pi_b
        logging.info(f"Range of policy ratios: {policy_ratios.min()}-{policy_ratios.max()}")
        target = reward * pi_e / pi_b
        target_normalised = (target - target.mean()) / max(target.std(), 1)
        reward_normalised = (reward - reward.mean()) / reward.std()
        self.w_y_nn.fit(
            X=reward_normalised.reshape(-1, 1),
            y=target_normalised.reshape(-1, 1),
        )
        self.h_y = (
            lambda x: self.w_y_nn.predict(
                (x.reshape(-1, 1) - reward.mean()) / reward.std()
            ).reshape(-1)
            * max(target.std(), 1)
            + target.mean()
        )
        h_y = self.h_y(reward).reshape(-1)
        logging.info(f"Range of h_values h(y): {h_y.min()}-{h_y.max()}")

    def _fit_w_y_binary(self, reward, pi_b, pi_e):
        policy_ratios = pi_e / pi_b
        self.policy_ratios = policy_ratios.reshape(-1)
        e_policy_ratios_y0 = policy_ratios[reward == 0].mean()
        e_policy_ratios_y1 = policy_ratios[reward == 1].mean()
        self.w_y = lambda x: (x == 0) * e_policy_ratios_y0 + (x == 1) * e_policy_ratios_y1
        self.w_y_weights = self.w_y(reward).reshape(-1)
        self.reward = reward.reshape(-1)

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if self.estimation_method == "alternative" and self.reward_type == "continuous":
            return self.h_y(reward)
        return reward * self.w_y(reward)

    def save_ratio_values(self, file_path: str):
        if self.estimation_method == "alternative" and self.reward_type == "continuous":
            raise NotImplementedError("Function not Implemented for alternative method")
        df = pd.DataFrame(
            data={
                "policy_ratios": self.policy_ratios,
                "weights_w_y": self.w_y_weights,
                "y": self.reward,
            }
        )
        df.to_csv(file_path, index=False)


class MIPS(BaseOffPolicyEstimator):
    """
    All credit for this code go to Yuta Saito and Thorsten Joachims.
    Original repository distirbuted under MIT license can be found
    in https://github.com/usaito/icml2022-mips
    """

    def _estimate_round_rewards(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_emb: np.ndarray,
        action_dist_b: np.ndarray,
        action_dist_e: np.ndarray,
        position: Optional[np.ndarray] = None,
        n_actions: Optional[int] = None,
        delta: float = 0.05,
        with_cnf: bool = False,
        **kwargs,
    ) -> np.ndarray:
        n = reward.shape[0]
        w_x_e = self._estimate_w_x_e(
            context=context,
            action=action,
            action_emb=action_emb,
            pi_e=action_dist_e[:, :, 0],
            pi_b=action_dist_b[:, :, 0],
            n_actions=n_actions,
        )

        if with_cnf:
            r_hat = reward * w_x_e
            cnf = np.sqrt(np.var(r_hat) / (n - 1))
            cnf *= stats.t.ppf(1.0 - (delta / 2), n - 1)

            return r_hat.mean(), cnf

        return reward * w_x_e

    def _estimate_w_x_e(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_emb: np.ndarray,
        pi_b: np.ndarray,
        pi_e: np.ndarray,
        n_actions: int,
    ) -> np.ndarray:
        n = action.shape[0]
        realized_actions = np.unique(action)
        w_x_a = pi_e / pi_b
        w_x_a = np.where(w_x_a < np.inf, w_x_a, 0)
        p_a_e_model = CategoricalNB()
        p_a_e_model.fit(action_emb, action)
        p_a_e = np.zeros((n, n_actions))
        p_a_e[:, realized_actions] = p_a_e_model.predict_proba(action_emb)
        w_x_e = (w_x_a * p_a_e).sum(1)

        return w_x_e

    def estimate_policy_value(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_emb: np.ndarray,
        action_dist_b: np.ndarray,
        action_dist_e: np.ndarray,
        n_actions: int,
        position: Optional[np.ndarray] = None,
        min_emb_dim: int = 1,
        feature_pruning: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action_emb, name="action_emb", expected_dim=2)
        check_array(array=action_dist_b, name="action_dist_b", expected_dim=3)
        check_array(array=action_dist_e, name="action_dist_e", expected_dim=3)

        if feature_pruning == "exact":
            return self._estimate_with_exact_pruning(
                context=context,
                reward=reward,
                action=action,
                action_emb=action_emb,
                action_dist_b=action_dist_b,
                action_dist_e=action_dist_e,
                n_actions=n_actions,
                position=position,
                min_emb_dim=min_emb_dim,
            )

        else:
            return self._estimate_round_rewards(
                context=context,
                reward=reward,
                action=action,
                action_emb=action_emb,
                action_dist_b=action_dist_b,
                action_dist_e=action_dist_e,
                n_actions=n_actions,
                position=position,
            ).mean()

    def _estimate_with_exact_pruning(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_emb: np.ndarray,
        action_dist_b: np.ndarray,
        action_dist_e: np.ndarray,
        n_actions: int,
        position: Optional[np.ndarray] = None,
        min_emb_dim: int = 1,
    ) -> float:
        n_emb_dim = action_emb.shape[1]
        min_emb_dim = np.int32(np.minimum(n_emb_dim, min_emb_dim))
        theta_list, cnf_list = [], []
        feat_list, C = np.arange(n_emb_dim), np.sqrt(6) - 1
        for i in np.arange(n_emb_dim, min_emb_dim - 1, -1):
            comb_list = list(itertools.combinations(feat_list, i))
            theta_list_, cnf_list_ = [], []
            for comb in comb_list:
                theta, cnf = self._estimate_round_rewards(
                    context=context,
                    reward=reward,
                    action=action,
                    action_emb=action_emb[:, comb],
                    action_dist_b=action_dist_b,
                    action_dist_e=action_dist_e,
                    n_actions=n_actions,
                    with_cnf=True,
                )
                if len(theta_list) > 0:
                    theta_list_.append(theta), cnf_list_.append(cnf)
                else:
                    theta_list.append(theta), cnf_list.append(cnf)
                    continue

            idx_list = np.argsort(cnf_list_)[::-1]
            for idx in idx_list:
                theta_i, cnf_i = theta_list_[idx], cnf_list_[idx]
                theta_j, cnf_j = np.array(theta_list), np.array(cnf_list)
                if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
                    theta_list.append(theta_i), cnf_list.append(cnf_i)
                else:
                    return theta_j[-1]

        return theta_j[-1]

    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure."""
        return NotImplementedError


def run_ope(
    val_bandit_data: Dict,
    action_dist_val: np.ndarray,
    estimated_rewards: np.ndarray,
    estimated_rewards_mrdr: np.ndarray,
    estimators: dict,
    synthetic=False,
) -> np.ndarray:
    """
    All credit for this code go to Yuta Saito and Thorsten Joachims.
    Original repository distirbuted under MIT license can be found
    in https://github.com/usaito/icml2022-mips
    """
    # lambdas = [10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]
    # lambdas_sg = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
    all_estimators = {
        "IPS": IPS(estimator_name="IPS"),
        "DR": DR(estimator_name="DR"),
        "DM": DM(estimator_name="DM"),
        "MRDR": DR(estimator_name="MRDR"),
    }
    if estimators:
        estimators_list = [
            all_estimators[k]
            for k in estimators
            if not (k.startswith("CHIPS") or k.startswith("MIPS") or k == 'MR')
        ]
    else:
        estimators_list = list(all_estimators.values())

    ope = OffPolicyEvaluation(bandit_feedback=val_bandit_data, ope_estimators=estimators_list)

    estimated_rewards_dict = {
        "DR": estimated_rewards,
        "DM": estimated_rewards,
        "MRDR": estimated_rewards_mrdr,
    }
    policy_values = ope.estimate_policy_values(
        action_dist=action_dist_val,
        pi_b=val_bandit_data["pi_b"],
        estimated_rewards_by_reg_model=estimated_rewards_dict,
    )

    if not synthetic:
        if "MIPS (w/o SLOPE)" in estimators or len(estimators) == 0:
            mips_estimate = MIPS().estimate_policy_value(
                context=val_bandit_data["context"],
                reward=val_bandit_data["reward"],
                action=val_bandit_data["action"],
                action_emb=val_bandit_data["action_context"],
                action_dist_b=val_bandit_data["pi_b"],
                action_dist_e=action_dist_val,
                n_actions=val_bandit_data["n_actions"],
                feature_pruning="no",
            )
            policy_values["MIPS (w/o SLOPE)"] = mips_estimate
        if "MIPS (w/ SLOPE)" in estimators or len(estimators) == 0:
            mips_estimate_slope = MIPS().estimate_policy_value(
                context=val_bandit_data["context"],
                reward=val_bandit_data["reward"],
                action=val_bandit_data["action"],
                action_emb=val_bandit_data["action_context"],
                action_dist_b=val_bandit_data["pi_b"],
                action_dist_e=action_dist_val,
                n_actions=val_bandit_data["n_actions"],
                feature_pruning="exact",
            )

            policy_values["MIPS (w/ SLOPE)"] = mips_estimate_slope

    return policy_values


@dataclass
class ModifiedOpenBanditDataset(OpenBanditDataset):
    """
    All credit for this code go to Yuta Saito and Thorsten Joachims.
    Original repository distirbuted under MIT license can be found
    in https://github.com/usaito/icml2022-mips
    """

    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return int(self.action.max() + 1)

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset."""
        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(self.data.loc[:, user_cols], drop_first=True).values
        pos = DataFrame(self.position)
        self.action_context = (
            self.item_context.drop(columns=["item_id", "item_feature_0"], axis=1)
            .apply(LabelEncoder().fit_transform)
            .values
        )
        self.action_context = self.action_context[self.action]
        self.action_context = np.c_[self.action_context, pos]

        self.action = self.position * self.n_actions + self.action
        self.position = np.zeros_like(self.position)
        self.pscore /= 3

    def sample_bootstrap_bandit_feedback(
        self,
        sample_size: Optional[int] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ) -> BanditFeedback:
        if is_timeseries_split:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )[0]
        else:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )
        n_rounds = bandit_feedback["n_rounds"]
        if sample_size is None:
            sample_size = bandit_feedback["n_rounds"]
        else:
            check_scalar(
                sample_size,
                name="sample_size",
                target_type=(int),
                min_val=0,
                max_val=n_rounds,
            )
        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(np.arange(n_rounds), size=sample_size, replace=True)
        self.idx = bootstrap_idx
        for key_ in [
            "action",
            "position",
            "reward",
            "pscore",
            "context",
            "action_context",
        ]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        bandit_feedback["n_rounds"] = sample_size
        return bandit_feedback


@dataclass
class MultiClassToBanditReduction(BaseBanditDataset):
    """
    Adapted class for handling multi-class classification data as logged bandit data.
    # Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
    # Licensed under the Apache 2.0 License.
    Modifications:
        -added return of indexes of separation in split_train_eval.

    Original code can be found at https://github.com/st-tech/zr-obp/blob/master/obp/dataset/multiclass.py


    Note
    -----
    A machine learning classifier such as logistic regression is used to construct behavior and evaluation policies as follows.

        1. Split the original data into training (:math:`\\mathcal{D}_{\\mathrm{tr}}`) and evaluation (:math:`\\mathcal{D}_{\\mathrm{ev}}`) sets.
        2. Train classifiers on :math:`\\mathcal{D}_{\\mathrm{tr}}` and obtain base deterministic policies :math:`\\pi_{\\mathrm{det},b}` and :math:`\\pi_{\\mathrm{det},e}`.
        3. Construct behavior (:math:`\\pi_{b}`) and evaluation (:math:`\\pi_{e}`) policies based on :math:`\\pi_{\\mathrm{det},b}` and :math:`\\pi_{\\mathrm{det},e}` as

            .. math::

                \\pi_b (a|x) := \\alpha_b \\cdot \\pi_{\\mathrm{det},b} (a|x) + (1.0 - \\alpha_b) \\cdot \\pi_{u} (a|x)

            .. math::

                \\pi_e (a|x) := \\alpha_e \\cdot \\pi_{\\mathrm{det},e} (a|x) + (1.0 - \\alpha_e) \\cdot \\pi_{u} (a|x)

            where :math:`\\pi_{u}` is a uniform random policy and :math:`\\alpha_b` and :math:`\\alpha_e` are given by the user.

        4. Measure the accuracy of the evaluation policy on :math:`\\mathcal{D}_{\\mathrm{ev}}` with its fully observed rewards and use it as the evaluation policy's ground truth policy value.

        5. Using :math:`\\mathcal{D}_{\\mathrm{ev}}`, an estimator :math:`\\hat{V}` estimates the policy value of the evaluation policy, i.e.,

            .. math::

                V(\\pi_e) \\approx \\hat{V} (\\pi_e; \\mathcal{D}_{\\mathrm{ev}})

        6. Evaluate the estimation performance of :math:`\\hat{V}` by comparing its estimate with the ground-truth policy value.

    Parameters
    -----------
    X: array-like, shape (n_rounds,n_features)
        Training vector of the original multi-class classification data,
        where `n_rounds` is the number of samples and `n_features` is the number of features.

    y: array-like, shape (n_rounds,)
        Target vector (relative to `X`) of the original multi-class classification data.

    base_classifier_b: ClassifierMixin
        Machine learning classifier used to construct a behavior policy.

    alpha_b: float, default=0.9
        Ratio of a uniform random policy when constructing a **behavior** policy.
        Must be in the [0, 1) interval to make the behavior policy stochastic.

    n_deficient_actions: int, default=0
        Number of deficient actions having zero probability of being selected in the logged bandit data.
        If there are some deficient actions, the full/common support assumption is very likely to be violated,
        leading to some bias for IPW-type estimators. See  Sachdeva et al.(2020) for details.
        `n_deficient_actions` should be an integer smaller than `n_actions - 1` so that there exists at least one actions
        that have a positive probability of being selected by the behavior policy.

    dataset_name: str, default=None
        Name of the dataset.

    Examples
    ----------

    .. code-block:: python

        # evaluate the estimation performance of IPW using the `digits` data in sklearn
        >>> import numpy as np
        >>> from sklearn.datasets import load_digits
        >>> from sklearn.linear_model import LogisticRegression
        # import open bandit pipeline (obp)
        >>> from obp.dataset import MultiClassToBanditReduction
        >>> from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW

        # load raw digits data
        >>> X, y = load_digits(return_X_y=True)
        # convert the raw classification data into the logged bandit dataset
        >>> dataset = MultiClassToBanditReduction(
            X=X,
            y=y,
            base_classifier_b=LogisticRegression(random_state=12345),
            alpha_b=0.8,
            dataset_name="digits",
        )
        # split the original data into the training and evaluation sets
        >>> dataset.split_train_eval(eval_size=0.7, random_state=12345)
        # obtain logged bandit feedback generated by behavior policy
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(random_state=12345)
        >>> bandit_feedback
        {
            'n_actions': 10,
            'n_rounds': 1258,
            'context': array([[ 0.,  0.,  0., ..., 16.,  1.,  0.],
                    [ 0.,  0.,  7., ..., 16.,  3.,  0.],
                    [ 0.,  0., 12., ...,  8.,  0.,  0.],
                    ...,
                    [ 0.,  1., 13., ...,  8., 11.,  1.],
                    [ 0.,  0., 15., ...,  0.,  0.,  0.],
                    [ 0.,  0.,  4., ..., 15.,  3.,  0.]]),
            'action': array([6, 8, 5, ..., 2, 5, 9]),
            'reward': array([1., 1., 1., ..., 1., 1., 1.]),
            'position': None,
            'pscore': array([0.82, 0.82, 0.82, ..., 0.82, 0.82, 0.82])
        }

        # obtain action choice probabilities by an evaluation policy and its ground-truth policy value
        >>> action_dist = dataset.obtain_action_dist_by_eval_policy(
            base_classifier_e=LogisticRegression(C=100, random_state=12345),
            alpha_e=0.9,
        )
        >>> ground_truth = dataset.calc_ground_truth_policy_value(action_dist=action_dist)
        >>> ground_truth
        0.865643879173291

        # off-policy evaluation using IPW
        >>> ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
        >>> estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)
        >>> estimated_policy_value
        {'ipw': 0.8662705029276045}

        # evaluate the estimation performance (accuracy) of IPW by relative estimation error (relative-ee)
        >>> relative_estimation_errors = ope.evaluate_performance_of_estimators(
                ground_truth_policy_value=ground_truth,
                action_dist=action_dist,
            )
        >>> relative_estimation_errors
        {'ipw': 0.000723881690137968}

    References
    ------------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Noveen Sachdeva, Yi Su, and Thorsten Joachims.
    "Off-policy Bandits with Deficient Support.", 2020.

    """

    X: np.ndarray
    y: np.ndarray
    base_classifier_b: ClassifierMixin
    alpha_b: float = 0.8
    n_deficient_actions: int = 0
    dataset_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize Class."""
        if not is_classifier(self.base_classifier_b):
            raise ValueError("`base_classifier_b` must be a classifier")
        check_scalar(self.alpha_b, "alpha_b", float, min_val=0.0)
        check_scalar(
            self.n_deficient_actions,
            "n_deficient_actions",
            int,
            min_val=0,
            max_val=self.n_actions - 1,
        )
        if self.alpha_b >= 1.0:
            raise ValueError(f"`alpha_b`= {self.alpha_b}, must be < 1.0.")

        self.X, y = check_X_y(X=self.X, y=self.y, ensure_2d=True, multi_output=False)
        self.y = (rankdata(y, "dense") - 1).astype(int)  # re-index actions
        # fully observed labels (full bandit feedback)
        self.y_full = np.zeros((self.n_rounds, self.n_actions))
        self.y_full[np.arange(self.n_rounds), y] = 1

    @property
    def len_list(self) -> int:
        """Length of recommendation lists, slate size."""
        return 1

    @property
    def n_actions(self) -> int:
        """Number of actions (number of classes)."""
        return np.unique(self.y).shape[0]

    @property
    def n_rounds(self) -> int:
        """Number of samples in the original multi-class classification data."""
        return self.y.shape[0]

    def split_train_eval(
        self,
        eval_size: Union[int, float] = 0.25,
        random_state: Optional[int] = None,
    ) -> None:
        """Split the original data into the training (used for policy learning) and evaluation (used for OPE) sets.

        Parameters
        ----------
        eval_size: float or int, default=0.25
            If float, should be between 0.0 and 1.0 and represent the proportion of the data to include in the evaluation split.
            If int, represents the absolute number of test samples.

        random_state: int, default=None
            Controls the random seed in train-evaluation split.

        """
        (
            self.X_tr,
            self.X_ev,
            self.y_tr,
            self.y_ev,
            _,
            self.y_full_ev,
            _,
            self.x_e_idx,
        ) = train_test_split(
            self.X,
            self.y,
            self.y_full,
            np.arange(len(self.X)),
            test_size=eval_size,
            random_state=random_state,
        )
        idx = sample(range(len(self.X_tr)), min(1000, len(self.X_tr)))
        self.X_tr = self.X_tr[idx]
        self.y_tr = self.y_tr[idx]
        self.n_rounds_ev = self.X_ev.shape[0]

    def obtain_batch_bandit_feedback(
        self,
        random_state: Optional[int] = None,
    ) -> BanditFeedback:
        """Obtain batch logged bandit data, an evaluation policy, and its ground-truth policy value.

        Note
        -------
        Please call `self.split_train_eval()` before calling this method.

        Parameters
        -----------
        random_state: int, default=None
            Controls the random seed in sampling actions.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            bandit_feedback is logged bandit data generated from a multi-class classification dataset.

        """
        random_ = check_random_state(random_state)
        # train a base ML classifier
        base_clf_b = self.base_classifier_b
        base_clf_b.fit(X=self.X_tr, y=self.y_tr)
        preds = base_clf_b.predict(self.X_ev).astype(int)
        # construct a behavior policy
        pi_b = np.zeros((self.n_rounds_ev, self.n_actions))
        pi_b[:, :] = (1.0 - self.alpha_b) / self.n_actions
        pi_b[np.arange(self.n_rounds_ev), preds] = (
            self.alpha_b + (1.0 - self.alpha_b) / self.n_actions
        )
        if self.n_deficient_actions > 0:
            deficient_actions = np.argsort(
                random_.gumbel(size=(self.n_rounds_ev, self.n_actions)), axis=1
            )[:, ::-1][:, : self.n_deficient_actions]
            deficient_actions_idx = (
                np.tile(np.arange(self.n_rounds_ev), (self.n_deficient_actions, 1)).T,
                deficient_actions,
            )
            pi_b[deficient_actions_idx] = 0.0  # create some deficient actions
            pi_b /= pi_b.sum(1)[:, np.newaxis]  # re-normalize the probability distribution
        # sample actions and factual rewards
        actions = sample_action_fast(pi_b, random_state=random_state)
        rewards = self.y_full_ev[np.arange(self.n_rounds_ev), actions]

        return dict(
            n_actions=self.n_actions,
            n_rounds=self.n_rounds_ev,
            context=self.X_ev,
            action=actions,
            reward=rewards,
            position=None,  # position effect is not considered in classification data
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(self.n_rounds_ev), actions],
        )

    def obtain_action_dist_by_eval_policy(
        self, base_classifier_e: Optional[ClassifierMixin] = None, alpha_e: float = 1.0
    ) -> np.ndarray:
        """Obtain action choice probabilities by an evaluation policy.

        Parameters
        -----------
        base_classifier_e: ClassifierMixin, default=None
            Machine learning classifier used to construct a behavior policy.

        alpha_e: float, default=1.0
            Ratio of a uniform random policy when constructing an **evaluation** policy.
            Must be in the [0, 1] interval (evaluation policy can be deterministic).

        Returns
        ---------
        action_dist_by_eval_policy: array-like, shape (n_rounds_ev, n_actions, 1)
            `action_dist_by_eval_policy` is the action choice probabilities of the evaluation policy.
            where `n_rounds_ev` is the number of samples in the evaluation set given the current train-eval split.
            `n_actions` is the number of actions.

        """
        check_scalar(alpha_e, "alpha_e", float, min_val=0.0, max_val=1.0)
        # train a base ML classifier
        if base_classifier_e is None:
            base_clf_e = clone(self.base_classifier_b)
        else:
            assert is_classifier(base_classifier_e), "`base_classifier_e` must be a classifier"
            base_clf_e = base_classifier_e
        base_clf_e.fit(X=self.X_tr, y=self.y_tr)
        preds = base_clf_e.predict(self.X_ev).astype(int)
        # construct an evaluation policy
        pi_e = np.zeros((self.n_rounds_ev, self.n_actions))
        pi_e[:, :] = (1.0 - alpha_e) / self.n_actions
        pi_e[np.arange(self.n_rounds_ev), preds] = alpha_e + (1.0 - alpha_e) / self.n_actions
        return pi_e[:, :, np.newaxis]

    def calc_ground_truth_policy_value(self, action_dist: np.ndarray) -> float:
        """Calculate the ground-truth policy value of the given action distribution.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds_ev, n_actions, 1)
            Action distribution or action choice probabilities of a policy whose ground-truth is to be caliculated here.
            where `n_rounds_ev` is the number of samples in the evaluation set given the current train-eval split.
            `n_actions` is the number of actions.

        Returns
        ---------
        ground_truth_policy_value: float
            policy value of given action distribution (mostly evaluation policy).

        """
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if action_dist.shape[0] != self.n_rounds_ev:
            raise ValueError(
                "Expected `action_dist.shape[0] == self.n_rounds_ev`, but found it False"
            )
        return action_dist[np.arange(self.n_rounds_ev), self.y_ev].mean()


class MultiClassToBanditReductionAdapted(MultiClassToBanditReduction):
    """
    All credit for this code go to Yuta Saito and Thorsten Joachims.
    Original repository distirbuted under MIT license can be found
    in https://github.com/usaito/icml2022-mips
    """

    def obtain_batch_bandit_feedback(
        self,
        beta_behav=1,
        random_state: Optional[int] = None,
        use_raw_behaviour_policy=False,
    ):
        """Obtain batch logged bandit data for evluation and training folds, an evaluation policy, and its ground-truth policy value.

        Note
        -------
        Please call `self.split_train_eval()` before calling this method.

        Parameters
        -----------
        random_state: int, default=None
            Controls the random seed in sampling actions.

        use_raw_behaviour_policy: bool, default=False
            if True, uses the classifier softmax scores as behaviour policy

        Returns
        ---------
        bandit_feedback_ev: BanditFeedback, bandit_feedback_tr: BanditFeedback
            bandit_feedback_ev (bandit_feedback_tr) is logged bandit data generated from a multi-class classification dataset for evaluation (training) fold.

        """
        random_ = check_random_state(random_state)
        # train a base ML classifier
        base_clf_b = self.base_classifier_b
        base_clf_b.fit(X=self.X_tr, y=self.y_tr)
        preds = base_clf_b.predict(self.X_ev).astype(int)
        preds_tr = base_clf_b.predict(self.X_tr).astype(int)
        # construct a behavior policy
        if use_raw_behaviour_policy:
            beh_policy = PolicyWrapper(classifier=base_clf_b, beta=beta_behav)
            pi_b = beh_policy.predict_proba(self.X_ev)
            pi_b_tr = beh_policy.predict_proba(self.X_tr)
        else:
            pi_b = np.zeros((self.n_rounds_ev, self.n_actions))
            pi_b[:, :] = (1.0 - self.alpha_b) / self.n_actions
            pi_b[np.arange(self.n_rounds_ev), preds] = (
                self.alpha_b + (1.0 - self.alpha_b) / self.n_actions
            )
            pi_b_tr = np.zeros((self.X_tr.shape[0], self.n_actions))
            pi_b_tr[:, :] = (1.0 - self.alpha_b) / self.n_actions
            pi_b_tr[np.arange(self.X_tr.shape[0]), preds_tr] = (
                self.alpha_b + (1.0 - self.alpha_b) / self.n_actions
            )
        if self.n_deficient_actions > 0:
            deficient_actions = np.argsort(
                random_.gumbel(size=(self.n_rounds_ev, self.n_actions)), axis=1
            )[:, ::-1][:, : self.n_deficient_actions]
            deficient_actions_idx = (
                np.tile(np.arange(self.n_rounds_ev), (self.n_deficient_actions, 1)).T,
                deficient_actions,
            )
            pi_b[deficient_actions_idx] = 0.0  # create some deficient actions
            pi_b /= pi_b.sum(1)[:, np.newaxis]  # re-normalize the probability distribution

            deficient_actions_tr = np.argsort(
                random_.gumbel(size=(self.X_tr.shape[0], self.n_actions)), axis=1
            )[:, ::-1][:, : self.n_deficient_actions]
            deficient_actions_idx_tr = (
                np.tile(np.arange(self.X_tr.shape[0]), (self.deficient_actions_tr, 1)).T,
                deficient_actions_tr,
            )
            pi_b_tr[deficient_actions_idx_tr] = 0.0  # create some deficient actions
            pi_b_tr /= pi_b_tr.sum(1)[:, np.newaxis]  # re-normalize the probability distribution
        # sample actions and factual rewards
        actions = sample_action_fast(pi_b, random_state=random_state)
        rewards = self.y_full_ev[np.arange(self.n_rounds_ev), actions]
        actions_tr = sample_action_fast(pi_b_tr, random_state=random_state)
        y_full_tr = np.eye(self.n_actions)[self.y_tr]
        rewards_tr = y_full_tr[np.arange(self.X_tr.shape[0]), actions_tr]

        return dict(
            n_actions=self.n_actions,
            n_rounds=self.n_rounds_ev,
            context=self.X_ev,
            action=actions,
            reward=rewards,
            position=None,  # position effect is not considered in classification data
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(self.n_rounds_ev), actions],
        ), dict(
            n_actions=self.n_actions,
            n_rounds=self.X_tr.shape[0],
            context=self.X_tr,
            action=actions_tr,
            reward=rewards_tr,
            position=None,  # position effect is not considered in classification data
            pi_b=pi_b_tr[:, :, np.newaxis],
            pscore=pi_b_tr[np.arange(self.X_tr.shape[0]), actions_tr],
        )

    def obtain_action_dist_by_eval_policy(
        self, base_classifier_e: Optional[ClassifierMixin] = None, alpha_e: float = 1.0
    ):
        """Obtain action choice probabilities by an evaluation policy for both training and evaluation data.

        Parameters
        -----------
        base_classifier_e: ClassifierMixin, default=None
            Machine learning classifier used to construct a behavior policy.

        alpha_e: float, default=1.0
            Ratio of a uniform random policy when constructing an **evaluation** policy.
            Must be in the [0, 1] interval (evaluation policy can be deterministic).

        Returns
        ---------
        action_dist_by_eval_policy: array-like, shape (n_rounds_ev, n_actions, 1), action_dist_by_eval_policy_tr: array-like, shape (n_rounds_tr, n_actions, 1)
            `action_dist_by_eval_policy` is the action choice probabilities of the evaluation policy for evaluation fold.
            `action_dist_by_eval_policy_tr` is the action choice probabilities of the evaluation policy for training fold.
            where `n_rounds_ev` (`n_rounds_tr`) is the number of samples in the evaluation (training) set given the current train-eval split.
            `n_actions` is the number of actions.

        """
        check_scalar(alpha_e, "alpha_e", float, min_val=0.0, max_val=1.0)
        # train a base ML classifier
        if base_classifier_e is None:
            base_clf_e = self.base_classifier_b
        else:
            assert is_classifier(base_classifier_e), "`base_classifier_e` must be a classifier"
            base_clf_e = base_classifier_e
        preds = base_clf_e.predict(self.X_ev).astype(int)
        preds_tr = base_clf_e.predict(self.X_tr).astype(int)
        preds_full = base_clf_e.predict(self.X).astype(int)
        # construct an evaluation policy
        pi_e = np.zeros((self.n_rounds_ev, self.n_actions))
        pi_e[:, :] = (1.0 - alpha_e) / self.n_actions
        pi_e[np.arange(self.n_rounds_ev), preds] = alpha_e + (1.0 - alpha_e) / self.n_actions
        pi_e_tr = np.zeros((self.X_tr.shape[0], self.n_actions))
        pi_e_tr[:, :] = (1.0 - alpha_e) / (self.n_actions - 1)
        pi_e_tr[np.arange(self.X_tr.shape[0]), preds_tr] = alpha_e
        pi_e_full = np.zeros((self.X.shape[0], self.n_actions))
        pi_e_full[:, :] = (1.0 - alpha_e) / (self.n_actions - 1)
        pi_e_full[np.arange(self.X.shape[0]), preds_full] = alpha_e
        return (
            pi_e[:, :, np.newaxis],
            pi_e_tr[:, :, np.newaxis],
            pi_e_full[:, :, np.newaxis],
        )

    def calc_ground_truth_policy_value(self, action_dist: np.ndarray) -> float:
        """Calculate the ground-truth policy value of the given action distribution.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds_ev, n_actions, 1)
            Action distribution or action choice probabilities of a policy whose ground-truth is to be caliculated here.
            where `n_rounds_ev` is the number of samples in the evaluation set given the current train-eval split.
            `n_actions` is the number of actions.

        Returns
        ---------
        ground_truth_policy_value: float
            policy value of given action distribution (mostly evaluation policy).

        """
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if action_dist.shape[0] == self.n_rounds_ev:
            return action_dist[np.arange(self.n_rounds_ev), self.y_ev].mean()
        elif action_dist.shape[0] == self.n_rounds:
            return action_dist[np.arange(self.n_rounds), self.y].mean()
        else:
            raise ValueError(
                "Expected `action_dist.shape[0] == self.n_rounds_ev` or `action_dist.shape[0] == self.n_rounds`, but found it False"
            )


@dataclass
class MarginalizedRatioWeighting(InverseProbabilityWeighting):
    """
    All credit for this code go to Muhammad Faaiz Taufiq, Arnaud Doucet,
    Rob Cornish and Jean-Francois Ton. Original code can be found at
    https://github.com/faaizt/mr-ope
    """

    def __init__(
        self,
        reward,
        pi_b,
        pi_e,
        reward_type,
        estimation_method="default",
        use_estimated_pscore=False,
        estimator_name="mr",
        **kwargs,
    ):
        self.estimator_name = estimator_name
        self.estimation_method = estimation_method
        self.reward_type = reward_type
        self.use_estimated_pscore = use_estimated_pscore
        if reward_type == "binary":
            self._fit_w_y_binary(reward, pi_b, pi_e)
        elif estimation_method == "default":
            self._fit_w_y(reward, pi_b, pi_e)
        else:
            self._fit_w_y_alternative(reward, pi_b, pi_e)

    def _fit_w_y(self, reward, pi_b, pi_e):
        self.w_y_nn = MLPRegressor(
            hidden_layer_sizes=(512, 256, 32),
            verbose=True,
            alpha=0.06,
            max_iter=1000,
            early_stopping=False,
        )
        policy_ratios = pi_e / pi_b
        self.policy_ratios = policy_ratios.reshape(-1)
        self.reward = reward.reshape(-1)
        logging.info(f"Range of policy ratios: {policy_ratios.min()}-{policy_ratios.max()}")
        policy_ratios_normalised = (policy_ratios - policy_ratios.mean()) / max(
            policy_ratios.std(), 1
        )
        reward_normalised = (reward - reward.mean()) / reward.std()
        self.w_y_nn.fit(
            X=reward_normalised.reshape(-1, 1),
            y=policy_ratios_normalised.reshape(-1, 1),
        )
        self.w_y = (
            lambda x: self.w_y_nn.predict(
                (x.reshape(-1, 1) - reward.mean()) / reward.std()
            ).reshape(-1)
            * max(policy_ratios.std(), 1)
            + policy_ratios.mean()
        )
        self.w_y_weights = self.w_y(reward).reshape(-1)
        logging.info(f"Range of ratios w(y): {self.w_y_weights.min()}-{self.w_y_weights.max()}")

    def _fit_w_y_alternative(self, reward, pi_b, pi_e):
        self.w_y_nn = MLPRegressor(
            hidden_layer_sizes=(512, 256, 32),
            verbose=True,
            alpha=0.06,
            max_iter=1000,
            early_stopping=False,
        )
        policy_ratios = pi_e / pi_b
        logging.info(f"Range of policy ratios: {policy_ratios.min()}-{policy_ratios.max()}")
        target = reward * pi_e / pi_b
        target_normalised = (target - target.mean()) / max(target.std(), 1)
        reward_normalised = (reward - reward.mean()) / reward.std()
        self.w_y_nn.fit(
            X=reward_normalised.reshape(-1, 1),
            y=target_normalised.reshape(-1, 1),
        )
        self.h_y = (
            lambda x: self.w_y_nn.predict(
                (x.reshape(-1, 1) - reward.mean()) / reward.std()
            ).reshape(-1)
            * max(target.std(), 1)
            + target.mean()
        )
        h_y = self.h_y(reward).reshape(-1)
        logging.info(f"Range of h_values h(y): {h_y.min()}-{h_y.max()}")

    def _fit_w_y_binary(self, reward, pi_b, pi_e):
        policy_ratios = pi_e / pi_b
        self.policy_ratios = policy_ratios.reshape(-1)
        e_policy_ratios_y0 = policy_ratios[reward == 0].mean()
        e_policy_ratios_y1 = policy_ratios[reward == 1].mean()
        self.w_y = lambda x: (x == 0) * e_policy_ratios_y0 + (x == 1) * e_policy_ratios_y1
        self.w_y_weights = self.w_y(reward).reshape(-1)
        self.reward = reward.reshape(-1)

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if self.estimation_method == "alternative" and self.reward_type == "continuous":
            return self.h_y(reward)
        return reward * self.w_y(reward)

    def save_ratio_values(self, file_path: str):
        if self.estimation_method == "alternative" and self.reward_type == "continuous":
            raise NotImplementedError("Function not Implemented for alternative method")
        df = pd.DataFrame(
            data={
                "policy_ratios": self.policy_ratios,
                "weights_w_y": self.w_y_weights,
                "y": self.reward,
            }
        )
        df.to_csv(file_path, index=False)


def create_ope_object(
    bandit_feedback_test: Dict,
    use_train_behav: bool,
    pi_b_scores: np.ndarray,
    pi_e_scores: np.ndarray,
    reward: np.ndarray,
    reward_type: str,
    estimation_method="default",
):
    """
    All credit for this code go to Muhammad Faaiz Taufiq, Arnaud Doucet,
    Rob Cornish and Jean-Francois Ton. Original code can be found at
    https://github.com/faaizt/mr-ope
    """

    lambdas = [10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback_test,
        ope_estimators=[
            IPW(),
            DM(),
            DR(),
            SwitchDR(
                lambdas=lambdas,
                tuning_method="slope",
                estimator_name="SwitchDR",
            ),
            DRos(
                lambdas=lambdas,
                tuning_method="slope",
                estimator_name="DRos",
            ),
            MarginalizedRatioWeighting(
                reward=reward,
                pi_b=pi_b_scores,
                pi_e=pi_e_scores,
                reward_type=reward_type,
                estimation_method=estimation_method,
                use_estimated_pscore=use_train_behav,
            ),
        ],
    )
    return ope
