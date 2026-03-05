from abc import ABC
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma

from .base import InfoTheoryEstimator, InfoTheoryMixin
from .utils import cov as covariance, buildvectors
from scipy.stats import f


class MVInfoTheoryEstimator(InfoTheoryMixin, InfoTheoryEstimator, ABC):
    """
    Base class for information theory estimators based on multivariate normal assumption.
    Implements these measures

    Entropy expressions and their estimators for multivariate distributions
    """

    def __init__(
        self, cov: Optional[np.ndarray] = None, mean: Optional[np.ndarray] = None
    ):
        self.cov = cov
        self.mu = mean


class MVNEntropy(MVInfoTheoryEstimator):
    """
    Computes the differential entropy of a given multivariate set of observations
    assuming that the probability distribution function for these observations is Gaussian
    Theorem 8.4.1 Cover & Thomas
    """

    def fit(self, X, y=None):
        if self.cov is None:
            self.cov = covariance(X)
        return self

    def score(self, X, y=None):
        C = self.cov
        # Entropy for the multivariate Gaussian case:
        n_dims = C.shape[0]
        # e.g., Barnett PRL 2009
        s, logdet = np.linalg.slogdet(C)
        e = 0.5 * (s * logdet) + 0.5 * n_dims * np.log(2 * np.pi * np.exp(1))
        return e


class MVLNEntropy(MVInfoTheoryEstimator):
    """
    Computes the differential entropy of a given multivariate set of observations
    assuming that the probability distribution function for these observations is LogNormal
    "Entropy Expressions and Their Estimators for Multivariate Distributions", Ahmed and Gokhale 1989
    Eq. 1.2
    """

    def fit(self, X, y=None):
        if self.cov is None:
            self.cov = covariance(X)
        if self.mu is None:
            self.mu = X.mean(axis=1)

        return self

    def score(self, X, y=None):
        return MVLNEntropy(cov=self.cov).fit(X).score(X, y) + self.mu.sum()


class MVParetoEntropy(MVInfoTheoryEstimator):
    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        # TODO equation 1.4 "Entropy Expressions and Their Estimators for Multivariate Distributions", Ahmed and Gokhale 1989
        raise NotImplementedError("To be implemented")
        pass


class MVExponentialEntropy(MVInfoTheoryEstimator):
    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        # TODO equation 1.5 "Entropy Expressions and Their Estimators for Multivariate Distributions", Ahmed and Gokhale 1989
        raise NotImplementedError("To be implemented")
        pass


class MVCondEntropy(MVInfoTheoryEstimator):
    def __init__(
        self,
        partial_cov: Optional[np.ndarray] = None,
        cov_x: Optional[np.ndarray] = None,
        cov_xy: Optional[np.ndarray] = None,
        cov_y: Optional[np.ndarray] = None,
    ):
        """
        Conditional entropy H(Y;X) = H(X,Y) - H(X)
        See Barnett PRL 2009, Eq.1
        Parameters
        ----------
        partial_cov
        cov_x
        cov_xy
        """
        self.partial_cov = partial_cov
        self.cov_x = cov_x
        self.cov_y = cov_y
        self.cov_xy = cov_xy

    def fit(self, X, y):
        # if self.cov_x is None:
        #     self.cov_x = cov(X)
        # if self.cov_y is None:
        #     self.cov_y = cov(y)
        # if self.cov_xy is None:
        #     self.cov_xy = cov(X,y)????

        # TODO sistemare questo calcolo dove si fornisce la matrice di cross-covarianza nxm dove n è la
        #  dimensione di X e m  la dimensione di y (nel esempio matlab m=1)
        # self.partial_cov = self.cov_x - (
        #     np.linalg.multi_dot(self.cov_xy, np.linalg.inv(self.cov_y), self.cov_xy.T)
        # )
        alpha = np.linalg.lstsq(X, y)[0]
        # computes the residuals epsilon
        epsilon = y - X @ alpha

        self.partial_cov = covariance(epsilon)
        return self

    def score(self, X, y=None):
        sign, log_det_part_cov = np.linalg.slogdet(self.partial_cov)
        n_dims = self.partial_cov.shape[0]
        return 0.5 * (sign * log_det_part_cov + n_dims * np.log(2 * np.pi * np.exp(1)))


class MVNMutualInformation(MVCondEntropy):
    def __init__(
        self,
    ):
        self.hy = None
        self.chxy = None

    def fit(self, X, y):
        self.hy = MVNEntropy().fit(X=y).score(X)
        self.chxy = MVCondEntropy().fit(X, y).score(X, y)
        return self

    def score(self, X, y=None):
        return self.hy - self.chxy


class KSGMutualInformation(InfoTheoryEstimator):
    """
    Kraskov-Stögbauer-Grassberger (KSG) estimator for mutual information.

    This is a k-nearest neighbor based estimator that's considered the gold standard
    for mutual information estimation in continuous spaces.

    References:
    - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
      Estimating mutual information. Physical review E, 69(6), 066138.
    """

    def __init__(self, k: int = 3, algorithm: int = 1, metric: str = "chebyshev"):
        """
        Initialize KSG estimator.

        Parameters
        ----------
        k : int, default=3
            Number of nearest neighbors to use for estimation.
            Should be >= 1, typically 3-5 works well.
        algorithm : int, default=1
            KSG algorithm variant (1 or 2). Algorithm 1 is recommended.
        metric : str, default='chebyshev'
            Distance metric to use. 'chebyshev' (max norm) is standard for KSG.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if algorithm not in [1, 2]:
            raise ValueError("algorithm must be 1 or 2")

        self.k = k
        self.algorithm = algorithm
        self.metric = metric
        self.mi_value_ = None

    def fit(self, X, y=None):
        """
        Fit the KSG estimator (no actual fitting needed for KSG).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            First variable data
        y : array-like of shape (n_samples, n_features_y)
            Second variable data

        Returns
        -------
        self : object
        """
        # KSG doesn't require fitting, but we validate inputs here
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")

        return self

    def score(self, X, y):
        """
        Compute mutual information between X and y using KSG estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            First variable data
        y : array-like of shape (n_samples, n_features_y)
            Second variable data

        Returns
        -------
        mi : float
            Estimated mutual information in nats
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        if n_samples != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if n_samples <= self.k:
            raise ValueError(f"Number of samples ({n_samples}) must be > k ({self.k})")

        # Combine X and y into joint space
        joint_data = np.column_stack([X, y])

        # For each point, find k-th nearest neighbor distance in joint space
        joint_distances = cdist(joint_data, joint_data, metric=self.metric)
        # Set diagonal to infinity to exclude self-matches
        np.fill_diagonal(joint_distances, np.inf)

        # Get k-th nearest neighbor distances
        kth_distances = np.partition(joint_distances, self.k - 1, axis=1)[:, self.k - 1]

        mi_sum = 0.0

        for i in range(n_samples):
            # Distance to k-th nearest neighbor in joint space
            epsilon = kth_distances[i]

            if epsilon == 0:
                # Handle degenerate case where points are identical
                continue

            # Count neighbors within epsilon distance in marginal spaces
            if self.algorithm == 1:
                # Algorithm 1: use strict inequality
                nx = np.sum(cdist(X[i : i + 1], X, metric=self.metric)[0] < epsilon)
                ny = np.sum(cdist(y[i : i + 1], y, metric=self.metric)[0] < epsilon)
            else:
                # Algorithm 2: use non-strict inequality
                nx = np.sum(cdist(X[i : i + 1], X, metric=self.metric)[0] <= epsilon)
                ny = np.sum(cdist(y[i : i + 1], y, metric=self.metric)[0] <= epsilon)

            # Avoid log(0) by ensuring nx, ny >= 1
            nx = max(1, nx)
            ny = max(1, ny)

            # KSG formula: digamma(k) + digamma(N) - digamma(nx) - digamma(ny)
            mi_sum += digamma(self.k) + digamma(n_samples) - digamma(nx) - digamma(ny)

        self.mi_value_ = mi_sum / n_samples
        return self.mi_value_


class KSGEntropy(InfoTheoryEstimator):
    """
    Kozachenko-Leonenko (KL) entropy estimator using k-nearest neighbors.

    This is the entropy estimator that forms the basis of the KSG mutual information estimator.

    References:
    - Kozachenko, L. F., & Leonenko, N. N. (1987).
      Sample estimate of the entropy of a random vector.
      Problemy Peredachi Informatsii, 23(2), 9-16.
    """

    def __init__(self, k: int = 3, metric: str = "chebyshev"):
        """
        Initialize KL entropy estimator.

        Parameters
        ----------
        k : int, default=3
            Number of nearest neighbors to use for estimation.
        metric : str, default='chebyshev'
            Distance metric to use.
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        self.k = k
        self.metric = metric
        self.entropy_value_ = None

    def fit(self, X, y=None):
        """
        Fit the KL estimator (no actual fitting needed).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data for entropy estimation
        y : ignored
            Not used, present for API consistency

        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self

    def score(self, X, y=None):
        """
        Compute differential entropy using KL estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data for entropy estimation
        y : ignored
            Not used, present for API consistency

        Returns
        -------
        entropy : float
            Estimated differential entropy in nats
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_dims = X.shape

        if n_samples <= self.k:
            raise ValueError(f"Number of samples ({n_samples}) must be > k ({self.k})")

        # Compute distances to k-th nearest neighbor for each point
        distances = cdist(X, X, metric=self.metric)
        np.fill_diagonal(distances, np.inf)  # Exclude self-matches

        # Get k-th nearest neighbor distances
        kth_distances = np.partition(distances, self.k - 1, axis=1)[:, self.k - 1]

        # Handle zero distances (degenerate cases)
        kth_distances = np.maximum(kth_distances, 1e-15)

        # KL entropy estimator formula
        # H = digamma(N) - digamma(k) + log(c_d) + (d/N) * sum(log(rho_k))
        # where c_d is the volume of unit ball in d dimensions

        # Volume of unit ball in d dimensions for Chebyshev metric
        if self.metric == "chebyshev":
            log_cd = n_dims * np.log(2)  # Volume of unit hypercube
        elif self.metric == "euclidean":
            # Volume of unit sphere in d dimensions
            log_cd = (n_dims / 2) * np.log(np.pi) - np.sum(
                np.log(np.arange(1, n_dims // 2 + 1))
            )
            if n_dims % 2 == 1:
                log_cd += np.log(2) * ((n_dims + 1) // 2) - np.sum(
                    np.log(np.arange(1, n_dims + 1, 2))
                )
        else:
            # Fallback approximation
            log_cd = n_dims * np.log(2)

        entropy = (
            digamma(n_samples)
            - digamma(self.k)
            + log_cd
            + n_dims * np.mean(np.log(kth_distances))
        )

        self.entropy_value_ = entropy
        return entropy


class MVKSGInfoTheoryEstimator(InfoTheoryEstimator):
    """
    Base class for multivariate KSG-based information theory estimators.

    This follows the same pattern as MVInfoTheoryEstimator but uses KSG methods
    instead of parametric assumptions.
    """

    def __init__(self, k: int = 3, algorithm: int = 1, metric: str = "chebyshev"):
        """
        Initialize multivariate KSG estimator.

        Parameters
        ----------
        k : int, default=3
            Number of nearest neighbors to use for estimation.
        algorithm : int, default=1
            KSG algorithm variant (1 or 2). Only used for MI estimation.
        metric : str, default='chebyshev'
            Distance metric to use.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if algorithm not in [1, 2]:
            raise ValueError("algorithm must be 1 or 2")

        self.k = k
        self.algorithm = algorithm
        self.metric = metric


class MVKSGCondEntropy(MVKSGInfoTheoryEstimator):
    """
    Multivariate conditional entropy H(Y|X) using KSG estimation.

    Uses the identity: H(Y|X) = H(X,Y) - H(X)

    References:
    - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
      Estimating mutual information. Physical review E, 69(6), 066138.
    """

    def __init__(self, k: int = 3, metric: str = "chebyshev"):
        super().__init__(k=k, metric=metric)
        self.h_xy_ = None
        self.h_x_ = None

    def fit(self, X, y):
        """
        Fit the conditional entropy estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Conditioning variables
        y : array-like of shape (n_samples, n_features_y)
            Target variables

        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        return self

    def score(self, X, y):
        """
        Compute conditional entropy H(Y|X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Conditioning variables
        y : array-like of shape (n_samples, n_features_y)
            Target variables

        Returns
        -------
        cond_entropy : float
            Conditional entropy in nats
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # H(Y|X) = H(X,Y) - H(X)
        entropy_estimator = KSGEntropy(k=self.k, metric=self.metric)

        # Combine X and y for joint entropy
        joint_data = np.column_stack([X, y])
        h_xy = entropy_estimator.fit(joint_data).score(joint_data)
        h_x = entropy_estimator.fit(X).score(X)

        self.h_xy_ = h_xy
        self.h_x_ = h_x

        return h_xy - h_x


class MVKSGCondMutualInformation(MVKSGInfoTheoryEstimator):
    """
    Conditional mutual information I(X;Y|Z) using KSG estimation.

    Uses the identity: I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)

    This is a key measure for understanding information relationships
    in multivariate systems with confounding variables.
    """

    def __init__(self, k: int = 3, metric: str = "chebyshev"):
        super().__init__(k=k, metric=metric)
        self.cmi_value_ = None

    def fit(self, X, y=None, Z=None):
        """
        Fit the conditional mutual information estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            First variable
        y : array-like of shape (n_samples, n_features_y)
            Second variable
        Z : array-like of shape (n_samples, n_features_z)
            Conditioning variable

        Returns
        -------
        self : object
        """
        if y is None or Z is None:
            raise ValueError("Both y and Z must be provided for conditional MI")

        X = np.asarray(X)
        y = np.asarray(y)
        Z = np.asarray(Z)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if not (X.shape[0] == y.shape[0] == Z.shape[0]):
            raise ValueError("X, y, and Z must have the same number of samples")

        return self

    def score(self, X, y, Z):
        """
        Compute conditional mutual information I(X;Y|Z).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            First variable
        y : array-like of shape (n_samples, n_features_y)
            Second variable
        Z : array-like of shape (n_samples, n_features_z)
            Conditioning variable

        Returns
        -------
        cmi : float
            Conditional mutual information in nats
        """
        X = np.asarray(X)
        y = np.asarray(y)
        Z = np.asarray(Z)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        # I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
        cond_entropy_estimator = MVKSGCondEntropy(k=self.k, metric=self.metric)

        h_x_given_z = cond_entropy_estimator.fit(Z, X).score(Z, X)
        h_y_given_z = cond_entropy_estimator.fit(Z, y).score(Z, y)

        # Combine X and y for joint conditional entropy
        xy_joint = np.column_stack([X, y])
        h_xy_given_z = cond_entropy_estimator.fit(Z, xy_joint).score(Z, xy_joint)

        cmi = h_x_given_z + h_y_given_z - h_xy_given_z
        self.cmi_value_ = cmi

        return cmi


class MVKSGTransferEntropy(MVKSGInfoTheoryEstimator):
    """
    Transfer entropy estimation using KSG method.

    Transfer entropy TE(X->Y) measures directed information transfer
    from X to Y, accounting for Y's own history.

    TE(X->Y) = I(Y_future; X_past | Y_past)

    References:
    - Schreiber, T. (2000). Measuring information transfer.
      Physical review letters, 85(2), 461-464.
    """

    def __init__(self, k: int = 3, metric: str = "chebyshev", lag: int = 1):
        """
        Initialize transfer entropy estimator.

        Parameters
        ----------
        k : int, default=3
            Number of nearest neighbors
        metric : str, default='chebyshev'
            Distance metric
        lag : int, default=1
            Time lag for transfer entropy computation
        """
        super().__init__(k=k, metric=metric)
        self.lag = lag
        self.te_value_ = None

    def fit(self, X, y=None):
        """
        Fit the transfer entropy estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Source time series
        y : array-like of shape (n_samples, n_features_y)
            Target time series

        Returns
        -------
        self : object
        """
        if y is None:
            raise ValueError("Both X and y must be provided for transfer entropy")

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if X.shape[0] <= self.lag:
            raise ValueError(f"Number of samples must be > lag ({self.lag})")

        return self

    def score(self, X, y):
        """
        Compute transfer entropy TE(X->Y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_x)
            Source time series
        y : array-like of shape (n_samples, n_features_y)
            Target time series

        Returns
        -------
        te : float
            Transfer entropy in nats
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Create lagged versions
        # X_past: X[t-lag]
        # Y_past: Y[t-lag]
        # Y_future: Y[t]

        n_samples = X.shape[0]
        X_past = X[: -self.lag]  # X[0:n-lag]
        Y_past = y[: -self.lag]  # Y[0:n-lag]
        Y_future = y[self.lag :]  # Y[lag:n]

        # TE(X->Y) = I(Y_future; X_past | Y_past)
        cmi_estimator = MVKSGCondMutualInformation(k=self.k, metric=self.metric)
        te = cmi_estimator.fit(Y_future, X_past, Y_past).score(Y_future, X_past, Y_past)

        self.te_value_ = te
        return te


class MVKSGPartialInformationDecomposition(MVKSGInfoTheoryEstimator):
    """
    Partial Information Decomposition (PID) using KSG estimation.

    Decomposes the information that two sources (X1, X2) provide about
    a target (Y) into:
    - Unique information from X1
    - Unique information from X2
    - Redundant information shared by X1 and X2
    - Synergistic information only available from X1 and X2 together

    References:
    - Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of
      multivariate information. arXiv preprint arXiv:1004.2515.
    """

    def __init__(self, k: int = 3, metric: str = "chebyshev"):
        super().__init__(k=k, metric=metric)
        self.unique_x1_ = None
        self.unique_x2_ = None
        self.redundant_ = None
        self.synergistic_ = None

    def fit(self, X1, X2=None, y=None):
        """
        Fit the PID estimator.

        Parameters
        ----------
        X1 : array-like of shape (n_samples, n_features_x1)
            First source variable
        X2 : array-like of shape (n_samples, n_features_x2)
            Second source variable
        y : array-like of shape (n_samples, n_features_y)
            Target variable

        Returns
        -------
        self : object
        """
        if X2 is None or y is None:
            raise ValueError("X1, X2, and y must all be provided for PID")

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        y = np.asarray(y)

        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if not (X1.shape[0] == X2.shape[0] == y.shape[0]):
            raise ValueError("X1, X2, and y must have the same number of samples")

        return self

    def score(self, X1, X2, y):
        """
        Compute partial information decomposition.

        Parameters
        ----------
        X1 : array-like of shape (n_samples, n_features_x1)
            First source variable
        X2 : array-like of shape (n_samples, n_features_x2)
            Second source variable
        y : array-like of shape (n_samples, n_features_y)
            Target variable

        Returns
        -------
        pid_dict : dict
            Dictionary containing 'unique_x1', 'unique_x2', 'redundant', 'synergistic'
        """
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        y = np.asarray(y)

        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Compute mutual informations
        mi_estimator = KSGMutualInformation(k=self.k, metric=self.metric)
        cmi_estimator = MVKSGCondMutualInformation(k=self.k, metric=self.metric)

        # I(X1; Y), I(X2; Y), I(X1,X2; Y)
        mi_x1_y = mi_estimator.fit(X1, y).score(X1, y)
        mi_x2_y = mi_estimator.fit(X2, y).score(X2, y)

        X12_joint = np.column_stack([X1, X2])
        mi_x12_y = mi_estimator.fit(X12_joint, y).score(X12_joint, y)

        # Conditional mutual informations
        # I(X1; Y | X2), I(X2; Y | X1)
        cmi_x1_y_given_x2 = cmi_estimator.fit(X1, y, X2).score(X1, y, X2)
        cmi_x2_y_given_x1 = cmi_estimator.fit(X2, y, X1).score(X2, y, X1)

        # PID components (using minimum redundancy approach)
        # Redundant: min(I(X1;Y), I(X2;Y))
        redundant = min(mi_x1_y, mi_x2_y)

        # Unique information
        unique_x1 = max(0, mi_x1_y - redundant)
        unique_x2 = max(0, mi_x2_y - redundant)

        # Synergistic: I(X1,X2;Y) - I(X1;Y) - I(X2;Y) + redundant
        synergistic = max(0, mi_x12_y - mi_x1_y - mi_x2_y + redundant)

        self.unique_x1_ = unique_x1
        self.unique_x2_ = unique_x2
        self.redundant_ = redundant
        self.synergistic_ = synergistic

        return {
            "unique_x1": unique_x1,
            "unique_x2": unique_x2,
            "redundant": redundant,
            "synergistic": synergistic,
            "total": mi_x12_y,
        }

from scipy.spatial import cKDTree

class KSGTransferEntropy(InfoTheoryEstimator):
    """
    KSG Bivariate Transfer Entropy Estimator.
    Maps to its_BTEknn.m
    """
    def __init__(self, driver_indices, target_indices, lags=1, k=3, metric='chebyshev'):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.lags = lags
        self.k = k
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        v_driver = [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target + v_driver)
        
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        
        y_present = B[:, 0:1]
        n_t = len(v_target)
        n_d = len(v_driver)
        
        y_past = B[:, 1:1+n_t]
        x_past = B[:, 1+n_t:1+n_t+n_d]
        
        M_y = y_present
        M_Y = y_past
        M_YZ = np.hstack([y_past, x_past])
        
        M_yY = np.hstack([M_y, M_Y])
        M_yYZ = np.hstack([M_y, M_YZ])
        
        p = np.inf if self.metric == 'chebyshev' else 2
        
        # 1. Neighbor search in highest dimension M_yYZ
        tree_yYZ = cKDTree(M_yYZ)
        # Note: k+1 because self is included
        distances, _ = tree_yYZ.query(M_yYZ, k=self.k + 1, p=p)
        dd = distances[:, -1]
        
        # Avoid 0 distances
        dd = np.maximum(dd, 1e-15)
        
        # 2. Range searches in lower dimensions
        def count_neighbors(M, dists):
            if M.shape[1] == 0:
                return np.full(N, N - 1)
            tree = cKDTree(M)
            counts = np.zeros(N)
            idx_list = tree.query_ball_point(M, r=dists, p=p)
            for i in range(N):
                pts = np.asarray(idx_list[i], dtype=int)
                # Match MATLAB range_search(..., past=0): exclude the query point itself.
                pts = pts[pts != i]
                if p == np.inf:
                    d_pts = np.max(np.abs(M[pts] - M[i]), axis=1)
                else:
                    d_pts = np.linalg.norm(M[pts] - M[i], axis=1)
                # Count points strictly < dists[i]
                c = len(pts) - np.sum(np.isclose(d_pts, dists[i], atol=1e-10))
                counts[i] = max(self.k - 1, c)
            return counts

        count_yY = count_neighbors(M_yY, dd)
        count_Y = count_neighbors(M_Y, dd)
        count_YZ = count_neighbors(M_YZ, dd)
        
        from scipy.special import digamma as psi
        TE = psi(self.k) + np.mean(psi(count_Y + 1) - psi(count_yY + 1) - psi(count_YZ + 1))
        
        self.transfer_entropy_ = TE
        
        dd2 = 2 * dd
        dd2 = dd2[dd2 > 0]
        if len(dd2) > 0:
            Hy_yz = -psi(self.k) + np.mean(psi(count_YZ + 1)) + np.mean(np.log(dd2))
        else:
            Hy_yz = np.nan
        self.conditional_entropy_ = Hy_yz
        
        return self


class KSGPartialTransferEntropy(InfoTheoryEstimator):
    """
    KSG Partial Transfer Entropy Estimator.
    Maps to its_PTEknn.m
    """
    def __init__(self, driver_indices, target_indices, conditioning_indices, lags=1, k=3, metric='chebyshev'):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.conditioning_indices = conditioning_indices
        self.lags = lags
        self.k = k
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        v_driver = [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        v_cond = [[self.conditioning_indices[0], l] for l in range(1, self.lags + 1)]
        
        V = np.array(v_target + v_driver + v_cond)
        
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        
        y_present = B[:, 0:1]
        n_t = len(v_target)
        n_d = len(v_driver)
        n_c = len(v_cond)
        
        y_past = B[:, 1:1+n_t]
        x_past = B[:, 1+n_t:1+n_t+n_d]
        z_past = B[:, 1+n_t+n_d:]
        
        M_y = y_present
        M_YZ = np.hstack([y_past, z_past])
        M_XYZ = np.hstack([y_past, x_past, z_past])
        
        M_yYZ = np.hstack([M_y, M_YZ])
        M_yXYZ = np.hstack([M_y, M_XYZ])
        
        p = np.inf if self.metric == 'chebyshev' else 2
        
        tree_B = cKDTree(B)
        distances, _ = tree_B.query(B, k=self.k + 1, p=p)
        dd = distances[:, -1]
        
        # Avoid 0 distances
        dd = np.maximum(dd, 1e-15)
        
        def count_neighbors(M, dists):
            if M.shape[1] == 0:
                return np.full(N, N - 1)
            tree = cKDTree(M)
            counts = np.zeros(N)
            idx_list = tree.query_ball_point(M, r=dists, p=p)
            for i in range(N):
                pts = np.asarray(idx_list[i], dtype=int)
                # Match MATLAB range_search(..., past=0): exclude the query point itself.
                pts = pts[pts != i]
                if p == np.inf:
                    d_pts = np.max(np.abs(M[pts] - M[i]), axis=1)
                else:
                    d_pts = np.linalg.norm(M[pts] - M[i], axis=1)
                c = len(pts) - np.sum(np.isclose(d_pts, dists[i], atol=1e-10))
                counts[i] = max(self.k - 1, c)
            return counts

        count_XYZ = count_neighbors(M_XYZ, dd)
        count_yYZ = count_neighbors(M_yYZ, dd)
        count_YZ = count_neighbors(M_YZ, dd)
        
        from scipy.special import digamma as psi
        # PTE formula: TE = I(Y_future; X_past | Y_past, Z_past)
        # TE = psi(k) + <psi(count_YZ + 1) - psi(count_yYZ + 1) - psi(count_XYZ + 1)>
        TE = psi(self.k) + np.mean(psi(count_YZ + 1) - psi(count_yYZ + 1) - psi(count_XYZ + 1))
        
        # NOTE: KSG estimator can be negative for very weak interactions due to sampling noise
        self.transfer_entropy_ = TE
        self.conditional_entropy_ = -psi(self.k) + np.mean(psi(count_XYZ + 1)) + np.mean(np.log(2 * dd))
        
        return self


class KernelTransferEntropy(InfoTheoryEstimator):
    """
    Kernel Bivariate Transfer Entropy Estimator.
    Maps to its_BTEker.m
    """
    def __init__(self, driver_indices, target_indices, lags=1, r=0.5, metric='chebyshev'):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.lags = lags
        self.r = r
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        v_driver = [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target + v_driver)
        
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        
        y_present = B[:, 0:1]
        n_t = len(v_target)
        n_d = len(v_driver)
        
        y_past = B[:, 1:1+n_t]
        x_past = B[:, 1+n_t:1+n_t+n_d]
        
        M_y = y_present
        M_Y = y_past
        M_YZ = np.hstack([y_past, x_past])
        
        M_yY = np.hstack([M_y, M_Y])
        M_yYZ = np.hstack([M_y, M_YZ])
        
        p = np.inf if self.metric == 'chebyshev' else 2
        
        def kernel_ce(M_y_and_M, M):
            # Same logic as its_CEker:
            # log ( count(M) / count(M_y_and_M) )
            # We count points within distance r, excluding self-matches.
            tree_full = cKDTree(M_y_and_M)
            pairs_full = tree_full.query_pairs(self.r, p=p)
            # Query pairs returns pairs (i, j) where i < j. 
            # Total matches is 2 * len(pairs)
            count_full = 2 * len(pairs_full)
            
            if M.shape[1] == 0:
                count_reduced = N * (N - 1)
            else:
                tree_reduced = cKDTree(M)
                pairs_reduced = tree_reduced.query_pairs(self.r, p=p)
                count_reduced = 2 * len(pairs_reduced)
            
            if count_full == 0 or count_reduced == 0:
                return np.nan
                
            return np.log(count_reduced / count_full)
            
        Hy_y = kernel_ce(M_yY, M_Y)
        Hy_yz = kernel_ce(M_yYZ, M_YZ)
        
        self.transfer_entropy_ = Hy_y - Hy_yz
        self.conditional_entropy_ = Hy_yz
        
        return self


class KernelPartialTransferEntropy(InfoTheoryEstimator):
    """
    Kernel Partial Transfer Entropy Estimator.
    Maps to its_PTEker.m
    """
    def __init__(self, driver_indices, target_indices, conditioning_indices, lags=1, r=0.5, metric='chebyshev'):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.conditioning_indices = conditioning_indices
        self.lags = lags
        self.r = r
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        v_driver = [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        v_cond = [[self.conditioning_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target + v_driver + v_cond)
        
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        
        y_present = B[:, 0:1]
        n_t = len(v_target)
        n_d = len(v_driver)
        n_c = len(v_cond)
        
        y_past = B[:, 1:1+n_t]
        x_past = B[:, 1+n_t:1+n_t+n_d]
        z_past = B[:, 1+n_t+n_d:]
        
        M_y = y_present
        M_YZ = np.hstack([y_past, z_past])
        M_XYZ = np.hstack([y_past, x_past, z_past])
        
        M_yYZ = np.hstack([M_y, M_YZ])
        M_yXYZ = np.hstack([M_y, M_XYZ])
        
        p = np.inf if self.metric == 'chebyshev' else 2
        
        def kernel_ce(M_y_and_M, M):
            tree_full = cKDTree(M_y_and_M)
            pairs_full = tree_full.query_pairs(self.r, p=p)
            count_full = 2 * len(pairs_full)
            
            if M.shape[1] == 0:
                count_reduced = N * (N - 1)
            else:
                tree_reduced = cKDTree(M)
                pairs_reduced = tree_reduced.query_pairs(self.r, p=p)
                count_reduced = 2 * len(pairs_reduced)
            
            if count_full == 0 or count_reduced == 0:
                return np.nan
                
            return np.log(count_reduced / count_full)
            
        Hy_yz = kernel_ce(M_yYZ, M_YZ)
        Hy_xyz = kernel_ce(M_yXYZ, M_XYZ)
        
        self.transfer_entropy_ = Hy_yz - Hy_xyz
        self.conditional_entropy_ = Hy_xyz
        
        return self


class KernelSelfEntropy(InfoTheoryEstimator):
    """
    Kernel Self Entropy Estimator (Information Storage).
    Maps to its_SEker.m
    """
    def __init__(self, target_indices, lags=1, r=0.5, metric='chebyshev'):
        self.target_indices = target_indices
        self.lags = lags
        self.r = r
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target)
        
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        
        y_present = B[:, 0:1]
        y_past = B[:, 1:]
        
        M_y = y_present
        M_Y = y_past
        M_yY = np.hstack([M_y, M_Y])
        
        p = np.inf if self.metric == 'chebyshev' else 2
        
        def kernel_e(M):
            if M.shape[1] == 0:
                return np.nan
            tree = cKDTree(M)
            pairs = tree.query_pairs(self.r, p=p)
            count = 2 * len(pairs)
            
            if count == 0:
                return np.nan
                
            return -np.log(count / (N * (N - 1)))
            
        def kernel_ce(M_y_and_M, M):
            tree_full = cKDTree(M_y_and_M)
            pairs_full = tree_full.query_pairs(self.r, p=p)
            count_full = 2 * len(pairs_full)
            
            if M.shape[1] == 0:
                count_reduced = N * (N - 1)
            else:
                tree_reduced = cKDTree(M)
                pairs_reduced = tree_reduced.query_pairs(self.r, p=p)
                count_reduced = 2 * len(pairs_reduced)
            
            if count_full == 0 or count_reduced == 0:
                return np.nan
                
            return np.log(count_reduced / count_full)
            
        Hy = kernel_e(M_y)
        Hy_y = kernel_ce(M_yY, M_Y)
        
        self.self_entropy_ = Hy - Hy_y
        
        return self
    """
    Kernel Bivariate Transfer Entropy Estimator.
    Maps to its_BTEker.m
    """
    def __init__(self, driver_indices, target_indices, lags=1, r=0.5, metric='chebyshev'):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.lags = lags
        self.r = r
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        v_driver = [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target + v_driver)
        
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        
        y_present = B[:, 0:1]
        n_t = len(v_target)
        n_d = len(v_driver)
        
        y_past = B[:, 1:1+n_t]
        x_past = B[:, 1+n_t:1+n_t+n_d]
        
        M_y = y_present
        M_Y = y_past
        M_YZ = np.hstack([y_past, x_past])
        
        M_yY = np.hstack([M_y, M_Y])
        M_yYZ = np.hstack([M_y, M_YZ])
        
        p = np.inf if self.metric == 'chebyshev' else 2
        
        def kernel_ce(M_y_and_M, M):
            # Same logic as its_CEker:
            # log ( count(M) / count(M_y_and_M) )
            # We count points within distance r, excluding self-matches.
            tree_full = cKDTree(M_y_and_M)
            pairs_full = tree_full.query_pairs(self.r, p=p)
            # Query pairs returns pairs (i, j) where i < j. 
            # Total matches is 2 * len(pairs)
            count_full = 2 * len(pairs_full)
            
            if M.shape[1] == 0:
                count_reduced = N * (N - 1)
            else:
                tree_reduced = cKDTree(M)
                pairs_reduced = tree_reduced.query_pairs(self.r, p=p)
                count_reduced = 2 * len(pairs_reduced)
            
            if count_full == 0 or count_reduced == 0:
                return np.nan
                
            return np.log(count_reduced / count_full)
            
        Hy_y = kernel_ce(M_yY, M_Y)
        Hy_yz = kernel_ce(M_yYZ, M_YZ)
        
        self.transfer_entropy_ = Hy_y - Hy_yz
        self.conditional_entropy_ = Hy_yz
        
        return self


class KernelPartialTransferEntropy(InfoTheoryEstimator):
    """
    Kernel Partial Transfer Entropy Estimator.
    Maps to its_PTEker.m
    """
    def __init__(self, driver_indices, target_indices, conditioning_indices, lags=1, r=0.5, metric='chebyshev'):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.conditioning_indices = conditioning_indices
        self.lags = lags
        self.r = r
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        v_driver = [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        v_cond = [[self.conditioning_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target + v_driver + v_cond)
        
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        
        y_present = B[:, 0:1]
        n_t = len(v_target)
        n_d = len(v_driver)
        n_c = len(v_cond)
        
        y_past = B[:, 1:1+n_t]
        x_past = B[:, 1+n_t:1+n_t+n_d]
        z_past = B[:, 1+n_t+n_d:]
        
        M_y = y_present
        M_YZ = np.hstack([y_past, z_past])
        M_XYZ = np.hstack([y_past, x_past, z_past])
        
        M_yYZ = np.hstack([M_y, M_YZ])
        M_yXYZ = np.hstack([M_y, M_XYZ])
        
        p = np.inf if self.metric == 'chebyshev' else 2
        
        def kernel_ce(M_y_and_M, M):
            tree_full = cKDTree(M_y_and_M)
            pairs_full = tree_full.query_pairs(self.r, p=p)
            count_full = 2 * len(pairs_full)
            
            if M.shape[1] == 0:
                count_reduced = N * (N - 1)
            else:
                tree_reduced = cKDTree(M)
                pairs_reduced = tree_reduced.query_pairs(self.r, p=p)
                count_reduced = 2 * len(pairs_reduced)
            
            if count_full == 0 or count_reduced == 0:
                return np.nan
                
            return np.log(count_reduced / count_full)
            
        Hy_yz = kernel_ce(M_yYZ, M_YZ)
        Hy_xyz = kernel_ce(M_yXYZ, M_XYZ)
        
        self.transfer_entropy_ = Hy_yz - Hy_xyz
        self.conditional_entropy_ = Hy_xyz
        
        return self


class KernelSelfEntropy(InfoTheoryEstimator):
    """
    Kernel Self Entropy Estimator (Information Storage).
    Maps to its_SEker.m
    """
    def __init__(self, target_indices, lags=1, r=0.5, metric='chebyshev'):
        self.target_indices = target_indices
        self.lags = lags
        self.r = r
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target)
        
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        
        y_present = B[:, 0:1]
        y_past = B[:, 1:]
        
        M_y = y_present
        M_Y = y_past
        M_yY = np.hstack([M_y, M_Y])
        
        p = np.inf if self.metric == 'chebyshev' else 2
        
        def kernel_e(M):
            if M.shape[1] == 0:
                return np.nan
            tree = cKDTree(M)
            pairs = tree.query_pairs(self.r, p=p)
            count = 2 * len(pairs)
            
            if count == 0:
                return np.nan
                
            return -np.log(count / (N * (N - 1)))
            
        def kernel_ce(M_y_and_M, M):
            tree_full = cKDTree(M_y_and_M)
            pairs_full = tree_full.query_pairs(self.r, p=p)
            count_full = 2 * len(pairs_full)
            
            if M.shape[1] == 0:
                count_reduced = N * (N - 1)
            else:
                tree_reduced = cKDTree(M)
                pairs_reduced = tree_reduced.query_pairs(self.r, p=p)
                count_reduced = 2 * len(pairs_reduced)
            
            if count_full == 0 or count_reduced == 0:
                return np.nan
                
            return np.log(count_reduced / count_full)
            
        Hy = kernel_e(M_y)
        Hy_y = kernel_ce(M_yY, M_Y)
        
        self.self_entropy_ = Hy - Hy_y
        
        return self
    """
    KSG Self Entropy Estimator (Information Storage).
    Maps to its_SEknn.m
    """
    def __init__(self, target_indices, lags=1, k=3, metric='chebyshev'):
        self.target_indices = target_indices
        self.lags = lags
        self.k = k
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target)
        
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        
        y_present = B[:, 0:1]
        y_past = B[:, 1:]
        
        M_y = y_present
        M_Y = y_past
        M_yY = np.hstack([M_y, M_Y])
        
        p = np.inf if self.metric == 'chebyshev' else 2
        
        tree_yY = cKDTree(M_yY)
        distances, _ = tree_yY.query(M_yY, k=self.k + 1, p=p)
        dd = distances[:, -1]
        
        # Add small value to avoid log(0) and division by 0
        dd = np.maximum(dd, 1e-15)
        
        def count_neighbors(M, dists):
            if M.shape[1] == 0:
                return np.full(N, N - 1)
            tree = cKDTree(M)
            counts = np.zeros(N)
            idx_list = tree.query_ball_point(M, r=dists, p=p)
            for i in range(N):
                pts = idx_list[i]
                if p == np.inf:
                    d_pts = np.max(np.abs(M[pts] - M[i]), axis=1)
                else:
                    d_pts = np.linalg.norm(M[pts] - M[i], axis=1)
                c = len(pts) - np.sum(np.isclose(d_pts, dists[i], atol=1e-10))
                counts[i] = max(self.k - 1, c)
            return counts

        count_Y = count_neighbors(M_Y, dd)
        count_y = count_neighbors(M_y, dd)
        
        from scipy.special import digamma as psi
        
        # Self Entropy calculation:
        # SE = I(y_present ; y_past) = H(y_present) + H(y_past) - H(y_present, y_past)
        SE = psi(self.k) + np.mean(psi(N) - psi(count_y + 1) - psi(count_Y + 1))
        
        self.self_entropy_ = SE
        
        return self


class GaussianTransferEntropy(InfoTheoryEstimator):
    """
    Gaussian Bivariate Transfer Entropy Estimator.
    Maps to its_BTElin.m
    """
    def __init__(self, driver_indices, target_indices, lags=1):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.lags = lags

    def _build_embeddings(self, X):
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        v_driver = [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target + v_driver)
        
        B = buildvectors(X, self.target_indices[0], V)
        
        y_present = B[:, 0:1]
        n_target_lags = len(v_target)
        y_past = B[:, 1:1+n_target_lags]
        x_past = B[:, 1+n_target_lags:]
        
        return y_present, y_past, x_past

    def fit(self, X, y=None):
        X = np.asarray(X)
        y_present, y_past, x_past = self._build_embeddings(X)
        
        res_restricted = self._ols_residuals(y_present, y_past)
        ce_restricted = self._gaussian_entropy_from_residuals(res_restricted)
        
        unrestricted_past = np.hstack([y_past, x_past])
        res_unrestricted = self._ols_residuals(y_present, unrestricted_past)
        ce_unrestricted = self._gaussian_entropy_from_residuals(res_unrestricted)
        
        self.transfer_entropy_ = ce_restricted - ce_unrestricted
        
        self.p_value_ = self._compute_f_test(
            res_restricted, res_unrestricted, 
            unrestricted_past.shape[1], y_past.shape[1]
        )
        return self

    def _ols_residuals(self, y, X):
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ coef

    def _gaussian_entropy_from_residuals(self, residuals):
        cov = np.atleast_2d(np.cov(residuals, rowvar=False))
        d = residuals.shape[1] if residuals.ndim > 1 else 1
        _, logdet = np.linalg.slogdet(cov)
        return 0.5 * logdet + 0.5 * d * np.log(2 * np.pi * np.exp(1))

    def _compute_f_test(self, res_r, res_u, n_unrestricted, n_restricted):
        rss_r = np.sum(res_r**2)
        rss_u = np.sum(res_u**2)
        n = res_r.shape[0] # Matches MATLAB length(Upr)
        num_restrictions = n_unrestricted - n_restricted
        df2 = n - n_unrestricted
        
        if num_restrictions == 0 or df2 == 0 or rss_u == 0:
            return 1.0
            
        f_stat = ((rss_r - rss_u) / num_restrictions) / (rss_u / df2)
        return f.sf(f_stat, num_restrictions, df2)


class GaussianPartialTransferEntropy(GaussianTransferEntropy):
    """
    Gaussian Partial Transfer Entropy Estimator.
    Maps to its_PTElin.m
    """
    def __init__(self, driver_indices, target_indices, conditioning_indices, lags=1):
        super().__init__(driver_indices, target_indices, lags)
        self.conditioning_indices = conditioning_indices

    def _build_embeddings_pte(self, X):
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        v_driver = [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        v_cond = [[self.conditioning_indices[0], l] for l in range(1, self.lags + 1)]
        
        V = np.array(v_target + v_driver + v_cond)
        
        B = buildvectors(X, self.target_indices[0], V)
        
        y_present = B[:, 0:1]
        n_t = len(v_target)
        n_d = len(v_driver)
        n_c = len(v_cond)
        
        y_past = B[:, 1:1+n_t]
        x_past = B[:, 1+n_t:1+n_t+n_d]
        z_past = B[:, 1+n_t+n_d:]
        
        return y_present, y_past, x_past, z_past

    def fit(self, X, y=None):
        X = np.asarray(X)
        y_present, y_past, x_past, z_past = self._build_embeddings_pte(X)
        
        # Restricted: y_present ~ [y_past, z_past]
        restricted_past = np.hstack([y_past, z_past])
        res_restricted = self._ols_residuals(y_present, restricted_past)
        ce_restricted = self._gaussian_entropy_from_residuals(res_restricted)
        
        # Unrestricted: y_present ~ [y_past, x_past, z_past]
        unrestricted_past = np.hstack([y_past, x_past, z_past])
        res_unrestricted = self._ols_residuals(y_present, unrestricted_past)
        ce_unrestricted = self._gaussian_entropy_from_residuals(res_unrestricted)
        
        self.transfer_entropy_ = ce_restricted - ce_unrestricted
        
        self.p_value_ = self._compute_f_test(
            res_restricted, res_unrestricted, 
            unrestricted_past.shape[1], restricted_past.shape[1]
        )
        return self


class GaussianSelfEntropy(GaussianTransferEntropy):
    """
    Gaussian Self Entropy Estimator (Information Storage).
    Maps to its_SElin.m
    """
    def __init__(self, target_indices, lags=1):
        super().__init__(driver_indices=[], target_indices=target_indices, lags=lags)

    def fit(self, X, y=None):
        X = np.asarray(X)
        # its_SElin is sensitive to mean and removes it
        X = X - np.mean(X, axis=0)
        
        target_data = X[:, self.target_indices[0]:self.target_indices[0]+1]
        cov_Y = np.atleast_2d(np.cov(target_data, rowvar=False))
        _, logdet = np.linalg.slogdet(cov_Y)
        Hy = 0.5 * logdet + 0.5 * 1 * np.log(2 * np.pi * np.exp(1))
        
        v_target = [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
        V = np.array(v_target)
        
        B = buildvectors(X, self.target_indices[0], V)
        y_present = B[:, 0:1]
        y_past = B[:, 1:]
        
        res_unrestricted = self._ols_residuals(y_present, y_past)
        Hy_y = self._gaussian_entropy_from_residuals(res_unrestricted)
        
        self.self_entropy_ = Hy - Hy_y
        
        # F-test
        # Null model (restricted): no regressors -> residuals = original target series (full length)
        # to exactly match Matlab's its_SElin.m logic: Uy=data(:,jj)
        res_restricted_null = target_data
        
        self.p_value_ = self._compute_f_test(
            res_restricted_null, res_unrestricted, 
            y_past.shape[1], 0
        )
        return self


# Clean final overrides for unstable intermediate blocks above.
class KSGSelfEntropy(InfoTheoryEstimator):
    """KSG self-entropy (information storage) estimator.

    Estimates information stored in the target process by combining a kNN search
    in joint space ``(Y_n, Y_n^-)`` with projected range counts in ``Y_n`` and
    ``Y_n^-`` subspaces.
    """

    def __init__(self, target_indices, lags=1, k=3, metric="chebyshev"):
        self.target_indices = target_indices
        self.lags = lags
        self.k = k
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        V = np.array([[self.target_indices[0], l] for l in range(1, self.lags + 1)])
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        y_present = B[:, 0:1]
        y_past = B[:, 1:]
        m_y = y_present
        m_Y = y_past
        m_yY = np.hstack([m_y, m_Y])
        p = np.inf if self.metric == "chebyshev" else 2
        tree = cKDTree(m_yY)
        distances, _ = tree.query(m_yY, k=self.k + 1, p=p)
        dd = np.maximum(distances[:, -1], 1e-15)

        def count_neighbors(M, dists):
            if M.shape[1] == 0:
                return np.full(N, N - 1.0)
            t = cKDTree(M)
            counts = np.zeros(N)
            idx_list = t.query_ball_point(M, r=dists, p=p)
            for i in range(N):
                pts = np.asarray(idx_list[i], dtype=int)
                # Match MATLAB range_search(..., past=0): exclude the query point itself.
                pts = pts[pts != i]
                if p == np.inf:
                    d_pts = np.max(np.abs(M[pts] - M[i]), axis=1)
                else:
                    d_pts = np.linalg.norm(M[pts] - M[i], axis=1)
                c = len(pts) - np.sum(np.isclose(d_pts, dists[i], atol=1e-10))
                counts[i] = max(self.k - 1, c)
            return counts

        count_y = count_neighbors(m_y, dd)
        count_Y = count_neighbors(m_Y, dd)
        from scipy.special import digamma as psi
        self.self_entropy_ = psi(self.k) + np.mean(psi(N) - psi(count_y + 1) - psi(count_Y + 1))
        self.conditional_entropy_ = -psi(self.k) + np.mean(psi(count_Y + 1)) + np.mean(np.log(2 * dd))
        return self


class KernelTransferEntropy(InfoTheoryEstimator):
    """Kernel bivariate transfer entropy estimator.

    Uses Heaviside-kernel counts with radius ``r`` to approximate conditional
    entropies in embedded spaces and computes ``TE = H(Y|Y^-) - H(Y|Y^-,X^-)``.
    """

    def __init__(self, driver_indices, target_indices, lags=1, r=0.5, metric="chebyshev"):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.lags = lags
        self.r = r
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        V = np.array(
            [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
            + [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
        )
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        y_present = B[:, 0:1]
        n_t = self.lags
        y_past = B[:, 1 : 1 + n_t]
        x_past = B[:, 1 + n_t :]
        m_yY = np.hstack([y_present, y_past])
        m_yYZ = np.hstack([y_present, y_past, x_past])
        p = np.inf if self.metric == "chebyshev" else 2

        def kernel_ce(full, reduced):
            t_full = cKDTree(full)
            c_full = 2 * len(t_full.query_pairs(self.r, p=p))
            if reduced.shape[1] == 0:
                c_red = N * (N - 1)
            else:
                t_red = cKDTree(reduced)
                c_red = 2 * len(t_red.query_pairs(self.r, p=p))
            if c_full == 0 or c_red == 0:
                return np.nan
            return np.log(c_red / c_full)

        hy_y = kernel_ce(m_yY, y_past)
        hy_yz = kernel_ce(m_yYZ, np.hstack([y_past, x_past]))
        self.transfer_entropy_ = hy_y - hy_yz
        self.conditional_entropy_ = hy_yz
        return self


class KernelPartialTransferEntropy(InfoTheoryEstimator):
    """Kernel partial transfer entropy estimator.

    Estimates ``PTE(X->Y|Z)`` by difference of kernel-based conditional
    entropies in ``(Y,Y^-,Z^-)`` and ``(Y,Y^-,X^-,Z^-)`` spaces.
    """

    def __init__(self, driver_indices, target_indices, conditioning_indices, lags=1, r=0.5, metric="chebyshev"):
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.conditioning_indices = conditioning_indices
        self.lags = lags
        self.r = r
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        V = np.array(
            [[self.target_indices[0], l] for l in range(1, self.lags + 1)]
            + [[self.driver_indices[0], l] for l in range(1, self.lags + 1)]
            + [[self.conditioning_indices[0], l] for l in range(1, self.lags + 1)]
        )
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        y_present = B[:, 0:1]
        n_t = self.lags
        n_d = self.lags
        y_past = B[:, 1 : 1 + n_t]
        x_past = B[:, 1 + n_t : 1 + n_t + n_d]
        z_past = B[:, 1 + n_t + n_d :]
        m_yYZ = np.hstack([y_present, y_past, z_past])
        m_yXYZ = np.hstack([y_present, y_past, x_past, z_past])
        p = np.inf if self.metric == "chebyshev" else 2

        def kernel_ce(full, reduced):
            t_full = cKDTree(full)
            c_full = 2 * len(t_full.query_pairs(self.r, p=p))
            if reduced.shape[1] == 0:
                c_red = N * (N - 1)
            else:
                t_red = cKDTree(reduced)
                c_red = 2 * len(t_red.query_pairs(self.r, p=p))
            if c_full == 0 or c_red == 0:
                return np.nan
            return np.log(c_red / c_full)

        hy_yz = kernel_ce(m_yYZ, np.hstack([y_past, z_past]))
        hy_xyz = kernel_ce(m_yXYZ, np.hstack([y_past, x_past, z_past]))
        self.transfer_entropy_ = hy_yz - hy_xyz
        self.conditional_entropy_ = hy_xyz
        return self


class KernelSelfEntropy(InfoTheoryEstimator):
    """Kernel self-entropy (information storage) estimator.

    Computes ``SE(Y)=H(Y)-H(Y|Y^-)`` using pair-count based entropy and
    conditional entropy approximations with radius ``r``.
    """

    def __init__(self, target_indices, lags=1, r=0.5, metric="chebyshev"):
        self.target_indices = target_indices
        self.lags = lags
        self.r = r
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X)
        V = np.array([[self.target_indices[0], l] for l in range(1, self.lags + 1)])
        B = buildvectors(X, self.target_indices[0], V)
        N = B.shape[0]
        y_present = B[:, 0:1]
        y_past = B[:, 1:]
        m_yY = np.hstack([y_present, y_past])
        p = np.inf if self.metric == "chebyshev" else 2

        def kernel_e(M):
            t = cKDTree(M)
            c = 2 * len(t.query_pairs(self.r, p=p))
            if c == 0:
                return np.nan
            return -np.log(c / (N * (N - 1)))

        def kernel_ce(full, reduced):
            t_full = cKDTree(full)
            c_full = 2 * len(t_full.query_pairs(self.r, p=p))
            if reduced.shape[1] == 0:
                c_red = N * (N - 1)
            else:
                t_red = cKDTree(reduced)
                c_red = 2 * len(t_red.query_pairs(self.r, p=p))
            if c_full == 0 or c_red == 0:
                return np.nan
            return np.log(c_red / c_full)

        hy = kernel_e(y_present)
        hy_y = kernel_ce(m_yY, y_past)
        self.self_entropy_ = hy - hy_y
        return self
