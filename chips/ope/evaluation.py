from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import obp
from obp.ope import RegressionModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from ope.legacy import run_ope, MarginalizedRatioWeighting
from ope.generation import generate_synthetic
from ope.estimators import BanditsConfig, IPS, DR, DM, CHIPS, MR, SNDR, DRoS, SNIPS
from ope.legacy import run_ope
from utils import generate_multiclass_policy
from ope.generation import generate_multiclass
from ope.legacy import MultiClassToBanditReductionAdapted, create_ope_object



def bayes_reward(alpha, beta, y, N):
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


def estimate_probs(
        x_num : int, 
        x_idx : np.ndarray, 
        A : np.ndarray, 
        a_num : int, 
        correction : Optional[bool] = True
    ) -> np.ndarray:
    """Estimates the probabilities of each action given a context from a sample.

    Parameters
    ----------
    x_num : int
        Number of possible contexts.
    x_idx : np.ndarray
        The indices of every context in the given sample.
    A : np.ndarray
        The actions taken for each context.
    a_num : int
        The total number of actions.
    correction : bool, optional
        Replace the 0 probabilities with 2e-16 for numerical stability, by default True

    Returns
    -------
    np.ndarray
        The estimated probabilities of each action given a context, dimension is x_num x a_num
    """
    XA = np.hstack([x_idx[:, np.newaxis].astype(int), A[:, np.newaxis].astype(int)])
    XA_unique, XA_counts = np.unique(XA, return_counts=True, axis=0)
    estimated_probs = np.zeros((x_num, a_num))
    estimated_probs[XA_unique[:, 0].astype(int), XA_unique[:, 1]] = XA_counts
    if correction:
        estimated_probs[estimated_probs == 0] = 2e-16
    sums = estimated_probs.sum(axis=1)[:, np.newaxis]
    sums[sums == 0] = 1
    estimated_probs = estimated_probs * 1 / sums
    return estimated_probs


def estimate_cluster_probs(
        probs : np.ndarray, 
        n_clusters : int, 
        cluster_idxs : np.ndarray, 
        correction : Optional[bool] = True
    ) -> np.ndarray:
    """Estimates the p(a_i | c_j) for each cluster and action. 

    Parameters
    ----------
    probs : np.ndarray
        The probabilities of each action given a context, p(a|x).
    n_clusters : int
        The number of clusters.
    cluster_idxs : np.ndarray
        The cluster index for each context.
    correction : Optional[bool], optional
        Replace the 0 probabilities with 2e-16 for numerical stability, by default True

    Returns
    -------
    np.ndarray
        The estimated probabilities of each action given a cluster, dimension is n_clusters x a_num
    """
    estimates = np.zeros((n_clusters, probs.shape[1]))
    u_c = np.unique(cluster_idxs)
    for i in u_c[u_c > -1].astype(int):
        estimates[i] = (p := probs[cluster_idxs == i].sum(axis=0)) / sum(p)
    if correction:
        estimates[estimates == 0] = 2e-16
    return estimates


def run_synthetic_experiment(
    x_dim : int,
    x_num : int,
    c_num : int,
    a_num : int,
    c_exp : float,
    cluster_rad : float,
    n_samples : int,
    runs : int,
    emp_c_num : int,
    beta : float,
    e_len : int,
    b_len : int,
    random_state : int,
    sigma : float,
    alpha_bayes : Optional[float] = 20,
    beta_bayes : Optional[float] = 20,
    estimators : Optional[list] = [],
    clustering_method : Optional[str] = "kmeans",
    def_actions : Optional[int] = 0,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Runs a synthetic experiment.

    Parameters
    ----------
    x_dim : int
        Dimension of the context features.
    x_num : int
        Number of different contexts.
    c_num : int
        Number of context clusters.
    a_num : int
        Number of actions.
    c_exp : float
        Context expansion factor, the generated centroids of the cluster contexts will be generated
        randomly within an n-dimensional ball of radius c_exp.
    cluster_rad : float
        The maximum distance that a context within a cluster can be from the cluster centroid.
    n_samples : int
        Number of samples to generate.
    runs : int
        Number of runs to repeat the experiment.
    emp_c_num : int
        Number of clusters to generate.
    beta : float
        The Beta factor, meassures how different the logging and evaluation policies are. The more negative
        the value, the more different the policies.
    e_len : int
        Number of samples of the generated evaluation dataset.
    b_len : int
        Number of samples of the generated logging dataset.
    random_state : int
        Random seed.
    sigma : float
        Context-specific behaviour deviation from cluster behaviour.
    alpha_bayes : Optional[float], optional
        The alpha parameter from the beta prior to use in MAP version of CHIPS, by default 20.
    beta_bayes : Optional[float], optional
        The beta parameter from the beta prior to use in MAP version of CHIPS, by default 20.
    estimators : Optional[list], optional
        A list of all estimators to be compared in the experiment, by default []
    clustering_method : Optional[str], optional
        The clustering method for empirically clustering when using CHIPS, by default "kmeans". Valid options are
        "kmeans", "gmm", "dbscan", "hdbscan", "birch", "agglomerative", "spectral", "affinity",
        "optics" and "mean_shift"
    def_actions : Optional[int], optional
        Number of deficient acitions, by default 0

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        A tuple of dictionaries containing a dictionary regarding the MSE, bias^2, and variance of the estimators
        in all the runs of the experiments and for every selected estimator.
    """
    all_estimators = {
        "DM": DM(),
        "DR": DR(),
        "IPS": IPS(),
        "CHIPS_bayes": CHIPS(use_bayes=True, alpha=alpha_bayes, beta=beta_bayes),
        "CHIPS_mean": CHIPS(use_bayes=False, alpha=alpha_bayes, beta=beta_bayes),
        "MR" : MR(),
        "DRoS" : DRoS(),
        "SNDR" : SNDR(),
        "SNIPS" : SNIPS(),
    }
    if estimators:
        all_estimators = {k: v for k, v in all_estimators.items() if k in estimators}
    out_estimators = {est: np.zeros(runs) for est in all_estimators}
    V_b = {est: np.zeros(runs) for est in all_estimators}
    variance = {est: None for est in all_estimators}
    bias_2 = {est: None for est in all_estimators}
    ve = np.zeros(runs)

    data = generate_synthetic(
        c_num,
        x_num,
        x_dim,
        c_exp,
        a_num,
        cluster_rad,
        e_len,
        b_len,
        beta,
        n_samples,
        n_samples,
        random_state=random_state,
        n_seeds=runs,
        sigma=sigma,
        def_actions=def_actions,
    )

    for pol, out in zip(range(runs), data):
        (_, A_e, _), (X_b, A_b, R_b), V_e, x_e_idx, x_b_idx, X = out

        # Generate the policies for the experiments
        pi_e = estimate_probs(x_num, x_e_idx, A_e, a_num)
        pi_b = estimate_probs(x_num, x_b_idx, A_b, a_num)

        # estimated rewards
        a_u = np.unique(A_b)
        reward_models = [
            RandomForestClassifier(n_estimators=10, max_samples=0.8, random_state=12345)
            for _ in np.arange(a_num)
        ]
        for i in a_u:
            mask = A_b == i
            reward_models[i].fit(X_b[mask], R_b[mask])

        config = BanditsConfig(
            X,
            x_b_idx,
            A_b,
            R_b,
            pi_e,
            pi_b,
            a_num,
            n_samples,
            emp_c_num,
            reward_models,
            alpha_bayes=alpha_bayes,
            clustering_method=clustering_method,
            def_actions=def_actions,
        )

        for est_name, est in all_estimators.items():
            V_b[est_name][pol] = est.run_ope(config)
            out_estimators[est_name][pol] = (V_b[est_name][pol] - V_e) ** 2

        ve[pol] = V_e
    base = out_estimators["IPS"].mean()
    for est in all_estimators:
        variance[est] = np.var(V_b[est])
        bias_2[est] = out_estimators[est].mean() - variance[est]
    return out_estimators, bias_2, variance


def run_real_experiment(
    sample_size : int,
    n_clusters : int,
    n_seeds : int,
    n_actions : int,
    X : np.ndarray,
    V_e : float,
    dataset_b : MultiClassToBanditReductionAdapted,
    policy_ts : obp.policy.BaseContextualPolicy,
    policy_ur : obp.policy.BaseContextualPolicy,
    x_b_idx_ : np.ndarray,
    bayes_alpha : Optional[float] = 20,
    bayes_beta : Optional[float] = 20,
    clustering_method : Optional[str] = "kmeans",
    estimators : Optional[list] = [],
    ) -> Dict[str, np.ndarray]:
    """Runs a real experiment.

    Parameters
    ----------
    sample_size : int
        Number of samples to generate
    n_clusters : int
        Number of clusters to estimate with.
    n_seeds : int
        Number of runs to repeat the experiment.
    n_actions : int
        Number of actions to consider.
    X : np.ndarray
        All possible contexts.
    V_e : float
        The expected reward of the evaluation policy.
    dataset_b : MultiClassToBanditReductionAdapted
        The dataset to use for the experiment.
    policy_ts : obp.policy.BaseContextualPolicy
        The target policy to use for the experiment.
    policy_ur : obp.policy.BaseContextualPolicy
        The logging policy to use for the experiment.
    x_b_idx_ : np.ndarray
        The indices of the contexts in the logging dataset.
    bayes_alpha : Optional[float], optional
        The alpha parameter for the beta prior, by default 20
    bayes_beta : Optional[float], optional
        The beta parameter for the beta prior, by default 20
    clustering_method : Optional[str], optional
        The clustering method to use for empirically clustering when using CHIPS, by default "kmeans"
        Other valid otions are "gmm", "dbscan", "hdbscan", "birch", "agglomerative", "spectral", "affinity",
        "optics" and "mean_shift".
    estimators : Optional[list], optional
        A list of all estimators to be compared in the experiment, by default []

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the MSE of the estimators in all the runs of the experiments and for every
        selected estimator. 
    """
    in_estimators = {
        "IPS": [None] * n_seeds,
        "CHIPS_bayes": [None] * n_seeds,
        "CHIPS_mean": [None] * n_seeds,
        "DM": [None] * n_seeds,
        "DR": [None] * n_seeds,
        "MRDR": [None] * n_seeds,
        "MIPS (w/o SLOPE)": [None] * n_seeds,
        "MIPS (w/ SLOPE)": [None] * n_seeds,
        "MR": [None] * n_seeds,
    }
    if estimators:
        in_estimators = {k: v for k, v in in_estimators.items() if k in estimators}
    i = 1
    for random_state in tqdm(range(n_seeds)):
        bandit_data_b = dataset_b.sample_bootstrap_bandit_feedback(
            sample_size=sample_size,
            random_state=random_state,
        )

        X_b, A_b, R_b = bandit_data_b["context"], bandit_data_b["action"], bandit_data_b["reward"]

        pi_e = policy_ts.compute_batch_action_dist(n_rounds=sample_size)
        p_e = pi_e.reshape(sample_size, n_actions, 1) / 3
        pi_b = policy_ur.compute_batch_action_dist(n_rounds=sample_size)
        p_b = pi_b.reshape(sample_size, n_actions, 1) / 3

        # estimate rewards

        policy_estimations = dict()
        regression_model = RegressionModel(
            n_actions=dataset_b.n_actions,
            base_model=RandomForestClassifier(n_estimators=10, max_samples=0.8, random_state=12345),
        )
        if len(set(("DR", "DM", "MRDR")) & set(estimators)) != 0 or len(estimators) == 0:
            estimated_rewards = regression_model.fit_predict(
                context=X_b,  # context; x
                action=A_b,  # action; a
                reward=R_b,  # reward; r
                n_folds=2,
                random_state=12345,
            )
        else:
            estimated_rewards = np.zeros((sample_size, n_actions, 1))

        bandit_data_b["pi_b"] = p_b
        policy_estimations = run_ope(
            bandit_data_b,
            p_e,
            estimated_rewards=estimated_rewards,
            estimated_rewards_mrdr=estimated_rewards,
            estimators=estimators,
        )

        x_b_idx = x_b_idx_[dataset_b.idx]

        pi_e = p_e.squeeze()[: X.shape[0]]
        pi_b = p_b.squeeze()[: X.shape[0]]

        config = BanditsConfig(
            X,
            x_b_idx,
            A_b,
            R_b,
            pi_e,
            pi_b,
            n_actions,
            sample_size,
            n_clusters,
            [],
            alpha_bayes=bayes_alpha,
            beta_bayes=bayes_beta,
            clustering_method=clustering_method,
        )
        policy_estimations["CHIPS_bayes"] = CHIPS(
            use_bayes=True, alpha=bayes_alpha, beta=bayes_beta
        ).run_ope(config)
        policy_estimations["CHIPS_mean"] = CHIPS(
            use_bayes=False, alpha=bayes_alpha, beta=bayes_beta
        ).run_ope(config)
        
        # Added to compare taufiq
        policy_estimations["MR"] = MarginalizedRatioWeighting(
            reward = R_b,
            pi_b = pi_b[x_b_idx_][A_b],
            pi_e = pi_e[x_b_idx_][A_b],
            reward_type="binary",
            use_estimated_pscore=True
        ).estimate_policy_value(R_b, A_b, p_b, estimated_pscore=bandit_data_b['pscore'])

        for est in in_estimators:
            in_estimators[est][random_state] = (policy_estimations[est] - V_e) ** 2

    return in_estimators


def run_multiclass_experiment(
    alpha_b: float,
    alpha_e: float,
    runs: int,
    eval_size: int,
    source: str,
    emp_c_num: int,
    alpha_bayes: float,
    path: Optional[str] = "",
    id: Optional[int] = -1,
    target_col: Optional[int] = 0,
    random_state: Optional[int] = 12345,
    ) -> Dict[str, np.ndarray]:
    """
    Runs a multiclass experiment to evaluate various off-policy estimators from
    multi-class bandit feedback data transformed from given data using the method described
    in Dudik et al. 2013.
    
    Parameters
    ----------
    alpha_b : float
        The alpha parameter for the behavior policy.
    alpha_e : float
        The alpha parameter for the evaluation policy.
    runs : int
        The number of runs to perform.
    eval_size : int
        The size of the evaluation set.
    source : str
        The source of the data.
    emp_c_num : int
        The empirical count number.
    alpha_bayes : float
        The alpha parameter for the Bayesian approach.
    path : Optional[str], default=""
        The path to the data.
    id : Optional[int], default=-1
        The ID of the dataset.
    target_col : Optional[int], default=0
        The target column index.
    random_state : Optional[int], default=12345
        The random state for reproducibility.
    
    Returns:
    --------
    Dict[str, np.ndarray]
        A dictionary containing the results of the experiment for each estimator.
    """
    
    estimators = ["IPS", "DR", "DM", "CHIPS_bayes", "MR", "SwitchDR", "DRos"]
    results = {est: np.zeros(runs) for est in estimators}
    X, y = generate_multiclass(source, random_state, path, id, target_col)

    for seed in tqdm(range(runs)):
        random_state = seed
        train_policy_classifier = LogisticRegression(random_state=random_state, solver="liblinear")
        dataset = MultiClassToBanditReductionAdapted(
            X=X,
            y=y,
            base_classifier_b=train_policy_classifier,
            dataset_name="",
            alpha_b=alpha_b,
        )

        # split the original data into training and evaluation sets
        dataset.split_train_eval(eval_size=eval_size, random_state=seed)

        # get training and testing data
        testing_feedback, training_feedback = dataset.obtain_batch_bandit_feedback(
            use_raw_behaviour_policy=False, random_state=seed
        )

        # obtain action choice probabilities of an evaluation policy
        # we construct an evaluation policy using Random Forest and parameter `alpha_e`

        """
        The following commented out code defines an alternative target policy class
        """

        eval_policy_classifier = LogisticRegression(random_state=random_state, solver="liblinear")
        eval_policy_classifier.fit(X=dataset.X_tr, y=dataset.y_tr)
        (
            action_dist_test,
            action_dist_tr,
            _,
        ) = dataset.obtain_action_dist_by_eval_policy(
            base_classifier_e=eval_policy_classifier, alpha_e=alpha_e
        )

        pi_e_scores = action_dist_tr[
            np.arange(action_dist_tr.shape[0]), training_feedback["action"], 0
        ]

        # estimate the expected rewards by using an ML model (Logistic Regression here)
        # the estimated rewards are used by model-dependent estimators such as DM and DR
        regression_model = RandomForestClassifier(
            n_estimators=10,
            max_samples=0.8,
            max_depth=2,
            random_state=seed,
        )
        regression_model.fit(training_feedback["context"], dataset.y_tr)

        estimated_rewards_by_reg_model = np.expand_dims(
            regression_model.predict_proba(testing_feedback["context"]), axis=-1
        )
        r_hat = np.zeros((testing_feedback["context"].shape[0], testing_feedback["n_actions"], 1))
        r_hat[:, regression_model.classes_, :] = estimated_rewards_by_reg_model
        estimated_rewards_by_reg_model = r_hat
        pi_b_scores_test = testing_feedback["pscore"]
        pi_b_scores_train = training_feedback["pscore"]

        ope = create_ope_object(
            bandit_feedback_test=testing_feedback,
            use_train_behav=False,
            pi_b_scores=pi_b_scores_train,
            pi_e_scores=pi_e_scores,
            reward=training_feedback["reward"],
            reward_type="binary",
        )

        X_uni, indices = np.unique(X, return_inverse=True, axis=0)
        ground_truth = dataset.calc_ground_truth_policy_value(
            action_dist=action_dist_test,
        )

        x_b_idx = indices[dataset.x_e_idx]
        pi_e = generate_multiclass_policy(
            X_uni, testing_feedback["n_actions"], eval_policy_classifier, alpha_e
        )
        pi_b = generate_multiclass_policy(
            X_uni, testing_feedback["n_actions"], train_policy_classifier, alpha_b
        )

        data = (
            X_uni,
            x_b_idx,
            testing_feedback["action"],
            testing_feedback["reward"],
            pi_e,
            pi_b,
            testing_feedback["n_actions"],
            testing_feedback["n_rounds"],
            emp_c_num,
            None,
        )

        estimations, _ = ope.summarize_off_policy_estimates(
            estimated_pscore=pi_b_scores_test,
            action_dist=action_dist_test,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        base = 1  # ((estimations.loc['ipw']['estimated_policy_value'] - ground_truth) ** 2)
        results["CHIPS_bayes"][seed] = (
            (
                CHIPS(use_bayes=True).run_ope(
                    BanditsConfig(*data, alpha_bayes=alpha_bayes, beta_bayes=alpha_bayes)
                )
                - ground_truth
            )
            ** 2
        ) / base
        results["DM"][seed] = (
            (estimations.loc["dm"]["estimated_policy_value"] - ground_truth) ** 2
        ) / base
        results["DR"][seed] = (
            (estimations.loc["dr"]["estimated_policy_value"] - ground_truth) ** 2
        ) / base
        results["MR"][seed] = (
            (estimations.loc["mr"]["estimated_policy_value"] - ground_truth) ** 2
        ) / base
        results["SwitchDR"][seed] = (
            (estimations.loc["SwitchDR"]["estimated_policy_value"] - ground_truth) ** 2
        ) / base
        results["DRos"][seed] = (
            (estimations.loc["DRos"]["estimated_policy_value"] - ground_truth) ** 2
        ) / base
        results["IPS"][seed] = (
            (estimations.loc["ipw"]["estimated_policy_value"] - ground_truth) ** 2
        ) / base

    return results
