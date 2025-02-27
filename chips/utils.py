import os
import random

import numpy as np

def seed_everything(seed=42):
    """
    Seeds the environment for reproducibility.

    Parameters
    ----------
    seed : int, optional
        The random seed value used for Python's built-in random module,
        NumPy's random module, and the PYTHONHASHSEED environment variable.
    Returns
    -------
    None
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed=42)

def random_ball(num_points, dimension, radius=1, center=None):

    """
    Generates random points uniformly within a ball of the specified radius
    and dimension. If a center is provided, points will be generated around
    that center.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
    dimension : int
        The dimension of the space in which points are generated.
    radius : float, optional
        The radius of the ball in which points are uniformly generated.
    center : numpy.ndarray, optional
        The center of the ball. If None, a zero vector is used.

    Returns
    -------
    numpy.ndarray
        An array of shape (num_points, dimension) containing the generated points.
    """
    center = np.zeros(dimension) if center is None else center
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = np.random.random((num_points,1)) ** (1/dimension)
    return center + radius * random_directions.T * random_radii


def softmax(x):
    """
    Computes the softmax function for each row in x.

    Softmax is computed by exponentiating the values in x, subtracting
    the max for numerical stability, and normalizing by the sum of exponentials.

    Parameters
    ----------
    x : numpy.ndarray
        The input array of shape (n_samples, n_features).

    Returns
    -------
    numpy.ndarray
        The softmax probabilities of the same shape as x.
    """

    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator

def generate_probs(X_c, n_actions, sigma=5):
    """
    Generates probabilities for each action given context data.

    The function creates a random matrix of shape (X_c.shape[0], n_actions)
    with normally distributed values scaled by sigma, takes absolute values,
    and normalizes each row to sum up to 1.

    Parameters
    ----------
    X_c : numpy.ndarray
        The context data of shape (n_samples, n_features).
    n_actions : int
        The number of possible actions.
    sigma : float, optional
        The standard deviation for generating random values.

    Returns
    -------
    numpy.ndarray
        A probability matrix of shape (n_samples, n_actions), where each row
        sums to 1.
    """
    M = np.random.randn(X_c.shape[0], n_actions) * sigma
    return (probs := abs(M)) * 1/probs.sum(axis=1, keepdims=True)


def vectorized(prob_matrix, items):
    """
    Vectorized sampling based on a probability matrix.

    Takes cumulative sums of probability columns, compares them against
    random values, and selects corresponding actions from the provided items.

    Parameters
    ----------
    prob_matrix : numpy.ndarray
        A 2D array of shape (n_rows, n_cols) representing probabilities,
        where each column is a probability distribution.
    items : array_like
        The items (e.g., actions) to be sampled based on the probabilities.

    Returns
    -------
    numpy.ndarray
        A 1D array of selected items, with length corresponding to the columns
        of prob_matrix.
    """
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]

def generate_actions(probs, x_idx):
    """
    Samples actions for given context indices based on a probability matrix.

    The function extracts the probability rows for the specified indices,
    transposes them for vectorized sampling, and returns the sampled actions.

    Parameters
    ----------
    probs : numpy.ndarray
        A 2D array of shape (n_samples, n_actions) representing action
        probabilities for each sample.
    x_idx : array_like
        A list or array of context indices for which actions are sampled.

    Returns
    -------
    numpy.ndarray
        An array of shape (len(x_idx),) containing sampled actions for each
        context in x_idx.
    """
    sample_probs = probs[x_idx]
    choices = vectorized(sample_probs.T, np.arange(probs.shape[1]))
    return choices

def generate_multiclass_policy(X, n_actions, clf, alpha):
    """
    Generates a stochastic policy from a deterministic classifier prediction.

    For each sample in X, the classifier predicts an action, which is turned
    into a one-hot vector. This deterministic behavior is mixed with a uniform
    distribution across actions based on alpha.

    Parameters
    ----------
    X : numpy.ndarray
        The input data of shape (n_samples, n_features).
    n_actions : int
        Total number of possible actions.
    clf : object
        A classifier with a .predict() method returning an action for each row.
    alpha : float
        The mixing coefficient between the deterministic policy (one-hot) and
        the uniform distribution.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n_samples, n_actions) representing the mixed
        policy for each sample.
    """
    det_p = np.zeros((X.shape[0], n_actions))
    det_p[np.arange(len(det_p)), clf.predict(X)] = 1
    p_i = det_p * alpha + (1-alpha) * np.ones_like(det_p, dtype=np.float32) / det_p.shape[1]
    return p_i