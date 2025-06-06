from abc import ABC
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma

from .base import InfoTheoryEstimator, InfoTheoryMixin
from .utils import cov as covariance


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
