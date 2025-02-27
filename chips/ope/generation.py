from typing import Tuple, Optional, List, Dict

import numpy as np
from obp.policy import Random, BernoulliTS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml, load_digits

from ope.legacy import ModifiedOpenBanditDataset
from ope.estimators import BanditsConfig
from utils import *


def generate_synthetic(
    c_num : int,
    x_num : int,
    x_dim : int,
    c_exp : float,
    a_num : int,
    cluster_rad : float,
    e_len : int,
    b_len : int,
    beta : float,
    log_samples : int,
    eval_samples : int,
    random_state : Optional[int] = 0,
    n_seeds : Optional[int] = 1,
    sigma : Optional[float] = .5,
    def_actions : Optional[int] = 0,
    **kwargs,
    ) -> List[Tuple]:
    """Generates n synthetic dataset with the specified parameters.

    Parameters
    ----------
    c_num : int
        Number of context clusters.
    x_num : int
        Number of different contexts.
    x_dim : int
        Dimension of the context features.
    c_exp : float
        Context expansion factor, the generated centroids of the cluster contexts will be generated
        randomly within an n-dimensional ball of radius c_exp.
    a_num : int
        Number of actions.
    cluster_rad : float
        The maximum distance that a context within a cluster can be from the cluster centroid.
    e_len : int
        Number of samples of the generated evaluation dataset.
    b_len : int
        Number of samples of the generated logging dataset.
    beta : float
        The Beta factor, meassures how different the logging and evaluation policies are. The more negative
        the value, the more different the policies.
    log_samples : int
        Number of samples to generate using the logging policy.
    eval_samples : int
        Number of samples to generate using the evaluation policy.
    random_state : int
        Random seed.
    n_seeds : int
        Number of runs to repeat the experiment.
    sigma : float
        Context-specific behaviour deviation from cluster behaviour.
    def_actions : Optional[int], optional
        Number of deficient acitions, by default 0

    Returns
    -------
    List[Tuple]
        A list of tuples containing the generated datasets and the associated values.
    """

    seed_everything(random_state)

    # Generate cluster centers
    C = random_ball(c_num, x_dim, radius=c_exp)

    # Generate contexts per cluster
    p = (p := np.random.rand(c_num, 1)) / p.sum()  # prob of selecting each cluster for generation

    c_x = np.random.choice(np.arange(len(C)), x_num, p=p.squeeze())
    X = np.zeros((x_num, x_dim))
    # generate points for each cluster
    for k, c in enumerate(C):
        idx = c_x == k
        X[idx] = random_ball(sum(idx), x_dim, radius=cluster_rad, center=c)

    # Generate the two datasets with two different policies
    p_x = (p := np.random.rand(x_num, 1)) / p.sum()  # generate probability of observing context
    x_e_idx = np.random.choice(x_num, e_len, p=p_x.squeeze())
    x_b_idx = np.random.choice(x_num, b_len, p=p_x.squeeze())
    dataset_b = {"x": X[x_b_idx], "a": None, "r": None}
    dataset_e = {"x": X[x_e_idx], "a": None, "r": None}

    M = []


    for n in range(n_seeds):
        seed_everything(n)

        c_prob = np.random.randn(c_num, a_num)
        context_noise = np.random.randn(x_num, a_num) * sigma

        
        p_b = softmax(beta * (c_prob[c_x] + context_noise))
        p_e = softmax(c_prob[c_x] + context_noise)

        if def_actions != 0:
            idx = np.random.choice(a_num, def_actions, replace=False)
            p_b[:, idx] = 0 
            p_b = p_b / p_b.sum(axis=1, keepdims=True)
        

        dataset_e["a"] = generate_actions(p_e, x_e_idx)
        dataset_b["a"] = generate_actions(p_b, x_b_idx)

        # values of the related
        # probability of choosing each action in each context we ssociate the probability inversly proportional
        mods_e = np.mean(abs(X[x_e_idx]), axis=1) * 1 / c_exp
        mods_b = np.mean(abs(X[x_b_idx]), axis=1) * 1 / c_exp

        # values of the related
        # probability of choosing each action in each context we ssociate the probability inversly proportional
        dataset_e["r"] = (np.random.rand(e_len) <= (p_e[x_e_idx, dataset_e["a"]] * mods_e)).astype(
            float
        )
        dataset_b["r"] = (np.random.rand(b_len) <= (p_e[x_b_idx, dataset_b["a"]] * mods_b)).astype(
            float
        )

        idx_e = np.random.choice(e_len, 1_000_000)
        idx_b = np.random.choice(b_len, log_samples)

        V_e = dataset_e["r"].mean()
        # for the high probabilities in p_e we put extreme vals
        M.append(
            (
                (dataset_e[x][idx_e] for x in dataset_e),
                (dataset_b[x][idx_b] for x in dataset_b),
                V_e,
                x_e_idx[idx_e],
                x_b_idx[idx_b],
                X,
            )
        )
    return M


def generate_real(config : BanditsConfig) -> Dict:
    """Generates the data for the real-world experiment.

    Parameters
    ----------
    config : ope.estimators.BanditsConfig
        The configuration for the real experiment.

    Returns
    -------
    Dict
        A dictionary containing the generated data for the real-world experiment.
    """
    # configurations
    random_state = 0

    # define policies
    policy_ur = Random(
        n_actions=80,
        len_list=3,
        random_state=random_state,
    )
    policy_ts = BernoulliTS(
        n_actions=80,
        len_list=3,
        random_state=random_state,
        is_zozotown_prior=True,
        campaign="all",
    )

    # calc ground-truth policy value (on-policy)
    V_e = ModifiedOpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy="bts", campaign="all", data_path=config["ope_path"]
    )

    # define a dataset class
    dataset_b = ModifiedOpenBanditDataset(
        behavior_policy="random",
        data_path=config["ope_path"],
        campaign="all",
    )

    dataset_e = ModifiedOpenBanditDataset(
        behavior_policy="bts",
        data_path=config["ope_path"],
        campaign="all",
    )

    # joint dummies connection, and then calculate dummies, otherwise embeddings of the same user in 
    # the logging dataset would not correspond to a record of the same user in the evaluation dataset

    user_cols = dataset_b.data.columns.str.contains("user_feature")
    all_features = pd.concat([dataset_b.data.loc[:, user_cols], dataset_e.data.loc[:, user_cols]])
    dummy_features = pd.get_dummies(all_features, drop_first=True).values
    X, iidx = np.unique(
        dummy_features, return_inverse=True, axis=0
    )  # the contexts and the inverse indices
    dataset_b.context, dataset_e.context = (
        dummy_features[: dataset_b.data.shape[0]],
        dummy_features[dataset_b.data.shape[0] :],
    )
    x_b_idx_ = iidx[: dataset_b.data.shape[0]]
    ret = {
        "policy_ts": policy_ts,
        "policy_ur": policy_ur,
        "V_e": V_e,
        "dataset_b": dataset_b,
        "dataset_e": dataset_e,
        "x_b_idx_": x_b_idx_,
        "X": X,
    }
    return ret


def generate_multiclass(
        source : str, 
        random_state : int, 
        path : Optional[str] = "", 
        id : Optional[int] = -1, 
        target_col : Optional[int] = 0
    )-> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a multi-class dataset.

    Parameters
    ----------
    source : str
        The source of the dataset. Possible values are "local", "openml", or "digits".
    random_state : int
        The random seed used to ensure reproducible results.
    path : Optional[str], default=""
        The file path to the CSV file if source is "local".
    id : Optional[int], default=-1
        The data ID from OpenML if source is "openml".
    target_col : Optional[int], default=0
        The column index in the dataset used as the target.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - X : np.ndarray
            The feature matrix.
        - y : np.ndarray
            The label-encoded target vector.
    
    Raises
    ------
    ValueError
        If the specified parameters are invalid or not recognized.
    """
    seed_everything(random_state)
    if source == "local":
        df = pd.read_csv(path)
        X = df.iloc[:, target_col:].to_numpy()
        y = LabelEncoder().fit_transform(df.iloc[:, target_col].to_numpy())
    elif source == "openml" and id > -1:
        data = fetch_openml(data_id=id)
        X = data["data"].to_numpy()
        y = LabelEncoder().fit_transform(data["target"].to_numpy())
    elif source == "digits":
        X, y = load_digits(return_X_y=True)
    else:
        raise ValueError(
            """
            Invalid Parameters; check source={"local", "cifar100", "openml", "digits"}
            dir is a valid path to a csv file if source="local" or id > 0 if
            source="openml"
            """
        )
    return X, y
