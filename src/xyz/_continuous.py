from __future__ import annotations

from abc import ABC

import numpy as np
from scipy.spatial import cKDTree  # type: ignore
from scipy.spatial.distance import cdist
from scipy.special import digamma, gammaln
from scipy.stats import f, norm, rankdata
from sklearn.utils.validation import check_is_fitted

from .base import InfoTheoryEstimator, InfoTheoryMixin
from .preprocessing import as_2d_array, as_trial_array, build_te_observations
from .utils import cov as covariance


def _validate_pair_inputs(X, y) -> tuple[np.ndarray, np.ndarray]:
    X = as_2d_array(X)
    y = as_2d_array(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    return X, y


def _validate_triplet_inputs(X, y, Z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = _validate_pair_inputs(X, y)
    Z = as_2d_array(Z)
    if Z.shape[0] != X.shape[0]:
        raise ValueError("X, y and Z must have the same number of samples")
    return X, y, Z


def _chebyshev_or_euclidean_p(metric: str) -> float:
    return np.inf if metric == "chebyshev" else 2


def _fit_and_return(estimator, *args):
    estimator.fit(*args)
    return estimator.score(*args)


def _unit_ball_log_volume(n_dims: int, metric: str) -> float:
    if metric == "chebyshev":
        return n_dims * np.log(2.0)
    if metric == "euclidean":
        return (n_dims / 2.0) * np.log(np.pi) - gammaln(n_dims / 2.0 + 1.0)
    return n_dims * np.log(2.0)


def _count_neighbors_within_radius(
    data: np.ndarray,
    radii: np.ndarray,
    *,
    metric: str,
    trial_ids: np.ndarray | None = None,
) -> np.ndarray:
    if data.shape[1] == 0:
        if trial_ids is None:
            return np.full(data.shape[0], data.shape[0] - 1.0)
        counts = np.zeros(data.shape[0], dtype=float)
        for trial_id in np.unique(trial_ids):
            mask = trial_ids == trial_id
            counts[mask] = max(np.sum(mask) - 1, 0)
        return counts

    p = _chebyshev_or_euclidean_p(metric)
    tree = cKDTree(data)
    idx_list = tree.query_ball_point(data, r=radii, p=p)
    counts = np.zeros(data.shape[0], dtype=float)
    for i, pts in enumerate(idx_list):
        pts = np.asarray(pts, dtype=int)
        pts = pts[pts != i]
        if trial_ids is not None:
            pts = pts[trial_ids[pts] == trial_ids[i]]
        if pts.size == 0:
            counts[i] = 0.0
            continue
        if p == np.inf:
            d_pts = np.max(np.abs(data[pts] - data[i]), axis=1)
        else:
            d_pts = np.linalg.norm(data[pts] - data[i], axis=1)
        counts[i] = float(len(pts) - np.sum(np.isclose(d_pts, radii[i], atol=1e-10)))
    return counts


def _joint_knn_radius(
    data: np.ndarray, *, k: int, metric: str, trial_ids: np.ndarray | None = None
) -> np.ndarray:
    if data.shape[0] <= k:
        raise ValueError(f"Number of samples ({data.shape[0]}) must be > k ({k})")

    p = _chebyshev_or_euclidean_p(metric)
    tree = cKDTree(data)
    query_k = min(data.shape[0], k + 8)
    distances, indices = tree.query(data, k=query_k, p=p)

    if query_k == 1:
        raise ValueError("Not enough points for nearest-neighbor estimation")
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    radii = np.empty(data.shape[0], dtype=float)
    for i in range(data.shape[0]):
        valid = []
        for dist, idx in zip(distances[i], indices[i], strict=False):
            if idx == i:
                continue
            if trial_ids is not None and trial_ids[idx] != trial_ids[i]:
                continue
            valid.append(dist)
            if len(valid) == k:
                break
        if len(valid) < k:
            # Fall back to a global distance computation only in the rare case
            # where the sampled neighbors were dominated by other trials.
            dist_full = cdist(data[i : i + 1], data, metric=metric)[0]
            if trial_ids is not None:
                mask = trial_ids == trial_ids[i]
                dist_full = dist_full[mask]
            dist_full = dist_full[dist_full > 0]
            if dist_full.size < k:
                raise ValueError(
                    "Not enough within-trial neighbors for the requested k"
                )
            valid = np.partition(dist_full, k - 1)[:k]
        radii[i] = valid[-1]
    return np.maximum(radii, 1e-15)


def _prepare_entropy_cond_inputs(
    conditioning: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    conditioning = as_2d_array(conditioning)
    target = as_2d_array(target)
    if conditioning.shape[0] != target.shape[0]:
        raise ValueError("conditioning and target must have the same number of samples")
    return conditioning, target


def _count_points_within_radii(
    data: np.ndarray,
    radii: np.ndarray,
    *,
    metric: str,
    strict: bool = False,
) -> np.ndarray:
    if data.shape[0] == 0:
        return np.zeros(0, dtype=int)
    p = _chebyshev_or_euclidean_p(metric)
    query_radii = np.nextafter(radii, 0.0) if strict else radii
    tree = cKDTree(data)
    return np.asarray(
        tree.query_ball_point(data, r=query_radii, p=p, return_length=True), dtype=int
    )


def _gaussian_copula_transform(data: np.ndarray) -> np.ndarray:
    data = as_2d_array(data)
    if data.shape[1] == 0:
        return np.empty((data.shape[0], 0), dtype=float)
    n_samples = data.shape[0]
    transformed = np.empty_like(data, dtype=float)
    for col_idx in range(data.shape[1]):
        ranks = rankdata(data[:, col_idx], method="average")
        uniforms = (ranks - 0.5) / n_samples
        transformed[:, col_idx] = norm.ppf(uniforms)
    return transformed


def _kernel_entropy(data: np.ndarray, *, radius: float, metric: str) -> float:
    p = _chebyshev_or_euclidean_p(metric)
    tree = cKDTree(data)
    pairs = tree.query_pairs(radius, p=p)
    count = 2 * len(pairs)
    n_samples = data.shape[0]
    if count == 0:
        return np.nan
    return float(-np.log(count / (n_samples * (n_samples - 1))))


def _kernel_conditional_entropy(
    full: np.ndarray,
    reduced: np.ndarray,
    *,
    radius: float,
    metric: str,
    trial_ids: np.ndarray | None = None,
) -> float:
    p = _chebyshev_or_euclidean_p(metric)
    full_tree = cKDTree(full)
    if trial_ids is None:
        count_full = 2 * len(full_tree.query_pairs(radius, p=p))
    else:
        count_full = 0
        for i, j in full_tree.query_pairs(radius, p=p):
            if trial_ids[i] == trial_ids[j]:
                count_full += 2

    if reduced.shape[1] == 0:
        if trial_ids is None:
            n = full.shape[0]
            count_reduced = n * (n - 1)
        else:
            count_reduced = 0
            for trial_id in np.unique(trial_ids):
                n = np.sum(trial_ids == trial_id)
                count_reduced += n * (n - 1)
    else:
        reduced_tree = cKDTree(reduced)
        if trial_ids is None:
            count_reduced = 2 * len(reduced_tree.query_pairs(radius, p=p))
        else:
            count_reduced = 0
            for i, j in reduced_tree.query_pairs(radius, p=p):
                if trial_ids[i] == trial_ids[j]:
                    count_reduced += 2

    if count_full == 0 or count_reduced == 0:
        return np.nan
    return float(np.log(count_reduced / count_full))


class MVInfoTheoryEstimator(InfoTheoryMixin, InfoTheoryEstimator, ABC):
    """Base class for parametric multivariate entropy estimators."""


class MVNEntropy(MVInfoTheoryEstimator):
    """Differential entropy under a multivariate Gaussian assumption."""

    score_attr_ = "entropy_"

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = as_2d_array(X)
        self.covariance_ = covariance(X)
        n_dims = self.covariance_.shape[0]
        sign, logdet = np.linalg.slogdet(self.covariance_)
        self.entropy_ = 0.5 * sign * logdet + 0.5 * n_dims * np.log(2 * np.pi * np.e)
        return self


class MVLNEntropy(MVInfoTheoryEstimator):
    """Differential entropy for log-normal observations."""

    score_attr_ = "entropy_"

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = as_2d_array(X)
        if np.any(X <= 0):
            raise ValueError(
                "Log-normal entropy requires strictly positive observations"
            )
        log_X = np.log(X)
        gaussian_entropy = MVNEntropy().fit(log_X).entropy_
        self.mean_log_ = np.mean(log_X, axis=0)
        self.entropy_ = gaussian_entropy + float(np.sum(self.mean_log_))
        return self


class MVParetoEntropy(MVInfoTheoryEstimator):
    """Placeholder for a future multivariate Pareto entropy estimator."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        raise NotImplementedError("Multivariate Pareto entropy is not implemented")

    def score(self, X=None, y=None):
        raise NotImplementedError("Multivariate Pareto entropy is not implemented")


class MVExponentialEntropy(MVInfoTheoryEstimator):
    """Placeholder for a future multivariate exponential entropy estimator."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        raise NotImplementedError("Multivariate exponential entropy is not implemented")

    def score(self, X=None, y=None):
        raise NotImplementedError("Multivariate exponential entropy is not implemented")


class MVCondEntropy(MVInfoTheoryEstimator):
    """Conditional entropy ``H(Y|X)`` under a linear-Gaussian model."""

    score_attr_ = "conditional_entropy_"

    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = _validate_pair_inputs(X, y)
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coef
        self.coefficients_ = coef
        self.residuals_ = residuals
        self.partial_covariance_ = covariance(residuals)
        n_dims = self.partial_covariance_.shape[0]
        sign, logdet = np.linalg.slogdet(self.partial_covariance_)
        self.conditional_entropy_ = 0.5 * sign * logdet + 0.5 * n_dims * np.log(
            2 * np.pi * np.e
        )
        return self


class MVNMutualInformation(MVInfoTheoryEstimator):
    """Mutual information under a multivariate Gaussian assumption."""

    score_attr_ = "mutual_information_"

    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = _validate_pair_inputs(X, y)
        self.hy_ = MVNEntropy().fit(y).entropy_
        self.hy_given_x_ = MVCondEntropy().fit(X, y).conditional_entropy_
        self.mutual_information_ = self.hy_ - self.hy_given_x_
        return self


class GaussianCopulaMutualInformation(MVInfoTheoryEstimator):
    """Mutual information after a Gaussian-copula marginal transform."""

    score_attr_ = "mutual_information_"

    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = _validate_pair_inputs(X, y)
        X_gc = _gaussian_copula_transform(X)
        y_gc = _gaussian_copula_transform(y)
        self.mutual_information_ = (
            MVNMutualInformation().fit(X_gc, y_gc).mutual_information_
        )
        return self


class GaussianCopulaConditionalMutualInformation(MVInfoTheoryEstimator):
    """Conditional mutual information after a Gaussian-copula transform."""

    score_attr_ = "conditional_mutual_information_"

    def __init__(self):
        pass

    def fit(self, X, y, Z):
        X, y, Z = _validate_triplet_inputs(X, y, Z)
        X_gc = _gaussian_copula_transform(X)
        y_gc = _gaussian_copula_transform(y)
        Z_gc = _gaussian_copula_transform(Z)
        self.hy_given_z_ = MVCondEntropy().fit(Z_gc, y_gc).conditional_entropy_
        self.hy_given_xz_ = (
            MVCondEntropy()
            .fit(np.column_stack([X_gc, Z_gc]), y_gc)
            .conditional_entropy_
        )
        self.conditional_mutual_information_ = self.hy_given_z_ - self.hy_given_xz_
        return self


class KSGMutualInformation(InfoTheoryMixin, InfoTheoryEstimator):
    """Kraskov-Stoegbauer-Grassberger mutual information estimator."""

    score_attr_ = "mutual_information_"

    def __init__(self, k: int = 3, algorithm: int = 1, metric: str = "chebyshev"):
        if k < 1:
            raise ValueError("k must be >= 1")
        if algorithm not in {1, 2}:
            raise ValueError("algorithm must be 1 or 2")
        self.k = k
        self.algorithm = algorithm
        self.metric = metric

    def fit(self, X, y):
        X, y = _validate_pair_inputs(X, y)
        n_samples = X.shape[0]
        if n_samples <= self.k:
            raise ValueError(f"Number of samples ({n_samples}) must be > k ({self.k})")

        joint = np.column_stack([X, y])
        radii = _joint_knn_radius(joint, k=self.k, metric=self.metric)
        strict = self.algorithm == 1
        nx = _count_points_within_radii(X, radii, metric=self.metric, strict=strict)
        ny = _count_points_within_radii(y, radii, metric=self.metric, strict=strict)
        mi_terms = digamma(self.k) + digamma(n_samples) - digamma(nx) - digamma(ny)
        self.mutual_information_ = float(np.mean(mi_terms))
        return self


class KSGEntropy(InfoTheoryMixin, InfoTheoryEstimator):
    """Kozachenko-Leonenko differential entropy estimator."""

    score_attr_ = "entropy_"

    def __init__(self, k: int = 3, metric: str = "chebyshev"):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.metric = metric

    def fit(self, X, y=None):
        X = as_2d_array(X)
        n_samples, n_dims = X.shape
        if n_samples <= self.k:
            raise ValueError(f"Number of samples ({n_samples}) must be > k ({self.k})")
        kth = _joint_knn_radius(X, k=self.k, metric=self.metric)
        log_cd = _unit_ball_log_volume(n_dims, self.metric)
        self.entropy_ = (
            digamma(n_samples)
            - digamma(self.k)
            + log_cd
            + n_dims * np.mean(np.log(kth))
        )
        return self


class MVKSGInfoTheoryEstimator(InfoTheoryMixin, InfoTheoryEstimator, ABC):
    """Base class for multivariate KSG-based estimators."""

    def __init__(self, k: int = 3, metric: str = "chebyshev"):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.metric = metric


class MVKSGCondEntropy(MVKSGInfoTheoryEstimator):
    """Multivariate conditional entropy ``H(Y|X)`` via KSG identities."""

    score_attr_ = "conditional_entropy_"

    def fit(self, X, y):
        X, y = _prepare_entropy_cond_inputs(X, y)
        joint = np.column_stack([X, y])
        self.h_xy_ = KSGEntropy(k=self.k, metric=self.metric).fit(joint).entropy_
        self.h_x_ = KSGEntropy(k=self.k, metric=self.metric).fit(X).entropy_
        self.conditional_entropy_ = self.h_xy_ - self.h_x_
        return self


class DirectKSGConditionalMutualInformation(MVKSGInfoTheoryEstimator):
    """Direct kNN conditional mutual information estimator."""

    score_attr_ = "conditional_mutual_information_"

    def fit(self, X, y, Z):
        X, y, Z = _validate_triplet_inputs(X, y, Z)
        joint = np.column_stack([X, y, Z])
        xz = np.column_stack([X, Z])
        yz = np.column_stack([y, Z])
        radii = _joint_knn_radius(joint, k=self.k, metric=self.metric)
        count_z = _count_neighbors_within_radius(Z, radii, metric=self.metric)
        count_xz = _count_neighbors_within_radius(xz, radii, metric=self.metric)
        count_yz = _count_neighbors_within_radius(yz, radii, metric=self.metric)
        self.conditional_mutual_information_ = float(
            digamma(self.k)
            + np.mean(
                digamma(count_z + 1) - digamma(count_xz + 1) - digamma(count_yz + 1)
            )
        )
        return self


class MVKSGCondMutualInformation(MVKSGInfoTheoryEstimator):
    """Conditional mutual information ``I(X;Y|Z)`` via KSG identities."""

    score_attr_ = "conditional_mutual_information_"

    def fit(self, X, y, Z):
        X, y, Z = _validate_triplet_inputs(X, y, Z)
        h_x_given_z = (
            MVKSGCondEntropy(k=self.k, metric=self.metric)
            .fit(Z, X)
            .conditional_entropy_
        )
        h_y_given_z = (
            MVKSGCondEntropy(k=self.k, metric=self.metric)
            .fit(Z, y)
            .conditional_entropy_
        )
        h_xy_given_z = (
            MVKSGCondEntropy(k=self.k, metric=self.metric)
            .fit(Z, np.column_stack([X, y]))
            .conditional_entropy_
        )
        self.conditional_mutual_information_ = h_x_given_z + h_y_given_z - h_xy_given_z
        return self


class MVKSGTransferEntropy(MVKSGInfoTheoryEstimator):
    """Multivariate TE computed as conditional mutual information."""

    score_attr_ = "transfer_entropy_"

    def __init__(self, k: int = 3, metric: str = "chebyshev", lag: int = 1):
        super().__init__(k=k, metric=metric)
        self.lag = lag

    def fit(self, X, y):
        X, y = _validate_pair_inputs(X, y)
        if X.shape[0] <= self.lag:
            raise ValueError(f"Number of samples must be > lag ({self.lag})")
        X_past = X[: -self.lag]
        Y_past = y[: -self.lag]
        Y_future = y[self.lag :]
        self.transfer_entropy_ = (
            MVKSGCondMutualInformation(k=self.k, metric=self.metric)
            .fit(Y_future, X_past, Y_past)
            .conditional_mutual_information_
        )
        return self


class MVKSGPartialInformationDecomposition(MVKSGInfoTheoryEstimator):
    """A simple minimum-redundancy PID based on KSG MI estimates."""

    def fit(self, X1, X2, y):
        X1, X2 = _validate_pair_inputs(X1, X2)
        y = as_2d_array(y)
        if y.shape[0] != X1.shape[0]:
            raise ValueError("X1, X2 and y must have the same number of samples")

        mi_est = KSGMutualInformation(k=self.k, metric=self.metric)
        cmi_est = MVKSGCondMutualInformation(k=self.k, metric=self.metric)

        mi_x1_y = mi_est.fit(X1, y).mutual_information_
        mi_x2_y = mi_est.fit(X2, y).mutual_information_
        mi_x12_y = mi_est.fit(np.column_stack([X1, X2]), y).mutual_information_

        redundant = min(mi_x1_y, mi_x2_y)
        unique_x1 = max(0.0, mi_x1_y - redundant)
        unique_x2 = max(0.0, mi_x2_y - redundant)
        synergistic = max(0.0, mi_x12_y - mi_x1_y - mi_x2_y + redundant)

        self.pid_ = {
            "unique_x1": unique_x1,
            "unique_x2": unique_x2,
            "redundant": redundant,
            "synergistic": synergistic,
            "total": mi_x12_y,
            "cmi_x1_y_given_x2": cmi_est.fit(X1, y, X2).conditional_mutual_information_,
            "cmi_x2_y_given_x1": cmi_est.fit(X2, y, X1).conditional_mutual_information_,
        }
        return self

    def score(self, X1=None, X2=None, y=None):
        if X1 is not None:
            self.fit(X1, X2, y)
        check_is_fitted(self, "pid_")
        return dict(self.pid_)


class _TimeSeriesEstimator(InfoTheoryMixin, InfoTheoryEstimator, ABC):
    def __init__(self):
        pass

    @staticmethod
    def _embedding_parts(
        X,
        *,
        target_index: int,
        lags: int,
        tau: int = 1,
        delay: int = 1,
        driver_index: int | None = None,
        driver_indices: list[int] | None = None,
        conditioning_indices: list[int] | None = None,
        extra_conditioning: str | None = None,
    ) -> dict[str, np.ndarray]:
        return build_te_observations(
            X,
            target_index=target_index,
            lags=lags,
            tau=tau,
            delay=delay,
            driver_index=driver_index,
            driver_indices=driver_indices,
            conditioning_indices=conditioning_indices,
            extra_conditioning=extra_conditioning,
        )


class _KSGTEBase(_TimeSeriesEstimator):
    def __init__(
        self,
        *,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        k: int = 3,
        metric: str = "chebyshev",
        extra_conditioning: str | None = None,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.lags = lags
        self.tau = tau
        self.delay = delay
        self.k = k
        self.metric = metric
        self.extra_conditioning = extra_conditioning

    def _fit_conditional_measure(self, y_present, restricted, full, trial_ids):
        m_y_restricted = np.hstack([y_present, restricted])
        joint = np.hstack([y_present, full])
        radii = _joint_knn_radius(
            joint, k=self.k, metric=self.metric, trial_ids=trial_ids
        )
        count_restricted = _count_neighbors_within_radius(
            restricted, radii, metric=self.metric, trial_ids=trial_ids
        )
        count_y_restricted = _count_neighbors_within_radius(
            m_y_restricted, radii, metric=self.metric, trial_ids=trial_ids
        )
        count_full = _count_neighbors_within_radius(
            full, radii, metric=self.metric, trial_ids=trial_ids
        )
        te = digamma(self.k) + np.mean(
            digamma(count_restricted + 1)
            - digamma(count_y_restricted + 1)
            - digamma(count_full + 1)
        )
        cond_entropy = (
            -digamma(self.k)
            + np.mean(digamma(count_full + 1))
            + y_present.shape[1] * np.mean(np.log(2 * radii))
        )
        return float(te), float(cond_entropy)


class KSGTransferEntropy(_KSGTEBase):
    """KSG bivariate transfer entropy estimator."""

    score_attr_ = "transfer_entropy_"

    def __init__(
        self,
        driver_indices,
        target_indices,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        k: int = 3,
        metric: str = "chebyshev",
        extra_conditioning: str | None = None,
    ):
        super().__init__(
            lags=lags,
            tau=tau,
            delay=delay,
            k=k,
            metric=metric,
            extra_conditioning=extra_conditioning,
        )
        self.driver_indices = driver_indices
        self.target_indices = target_indices

    def fit(self, X, y=None):
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
            delay=self.delay,
            driver_indices=self.driver_indices,
            extra_conditioning=self.extra_conditioning,
        )
        restricted = np.hstack([parts["y_past"], parts["faes_current"]])
        full = np.hstack([parts["y_past"], parts["x_past"], parts["faes_current"]])
        self.transfer_entropy_, self.conditional_entropy_ = (
            self._fit_conditional_measure(
                parts["y_present"], restricted, full, parts["trial_ids"]
            )
        )
        return self


class KSGPartialTransferEntropy(_KSGTEBase):
    """KSG partial transfer entropy estimator."""

    score_attr_ = "transfer_entropy_"

    def __init__(
        self,
        driver_indices,
        target_indices,
        conditioning_indices,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        k: int = 3,
        metric: str = "chebyshev",
        extra_conditioning: str | None = None,
    ):
        super().__init__(
            lags=lags,
            tau=tau,
            delay=delay,
            k=k,
            metric=metric,
            extra_conditioning=extra_conditioning,
        )
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.conditioning_indices = conditioning_indices

    def fit(self, X, y=None):
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
            delay=self.delay,
            driver_indices=self.driver_indices,
            conditioning_indices=self.conditioning_indices,
            extra_conditioning=self.extra_conditioning,
        )
        restricted = np.hstack(
            [parts["y_past"], parts["z_past"], parts["faes_current"]]
        )
        full = np.hstack(
            [parts["y_past"], parts["x_past"], parts["z_past"], parts["faes_current"]]
        )
        self.transfer_entropy_, self.conditional_entropy_ = (
            self._fit_conditional_measure(
                parts["y_present"], restricted, full, parts["trial_ids"]
            )
        )
        return self


class KSGSelfEntropy(_KSGTEBase):
    """KSG information storage / self-entropy estimator."""

    score_attr_ = "self_entropy_"

    def __init__(
        self,
        target_indices,
        lags: int = 1,
        tau: int = 1,
        k: int = 3,
        metric: str = "chebyshev",
    ):
        super().__init__(lags=lags, tau=tau, delay=1, k=k, metric=metric)
        self.target_indices = target_indices

    def fit(self, X, y=None):
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
        )
        joint = np.hstack([parts["y_present"], parts["y_past"]])
        radii = _joint_knn_radius(
            joint, k=self.k, metric=self.metric, trial_ids=parts["trial_ids"]
        )
        count_y = _count_neighbors_within_radius(
            parts["y_present"], radii, metric=self.metric, trial_ids=parts["trial_ids"]
        )
        count_y_past = _count_neighbors_within_radius(
            parts["y_past"], radii, metric=self.metric, trial_ids=parts["trial_ids"]
        )
        n_effective = joint.shape[0]
        self.self_entropy_ = float(
            digamma(self.k)
            + np.mean(
                digamma(n_effective) - digamma(count_y + 1) - digamma(count_y_past + 1)
            )
        )
        self.conditional_entropy_ = float(
            -digamma(self.k)
            + np.mean(digamma(count_y_past + 1))
            + np.mean(np.log(2 * radii))
        )
        return self


class _KernelTEBase(_TimeSeriesEstimator):
    def __init__(
        self,
        *,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        r: float = 0.5,
        metric: str = "chebyshev",
        extra_conditioning: str | None = None,
    ):
        self.lags = lags
        self.tau = tau
        self.delay = delay
        self.r = r
        self.metric = metric
        self.extra_conditioning = extra_conditioning


class KernelTransferEntropy(_KernelTEBase):
    """Kernel bivariate transfer entropy estimator."""

    score_attr_ = "transfer_entropy_"

    def __init__(
        self,
        driver_indices,
        target_indices,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        r: float = 0.5,
        metric: str = "chebyshev",
        extra_conditioning: str | None = None,
    ):
        super().__init__(
            lags=lags,
            tau=tau,
            delay=delay,
            r=r,
            metric=metric,
            extra_conditioning=extra_conditioning,
        )
        self.driver_indices = driver_indices
        self.target_indices = target_indices

    def fit(self, X, y=None):
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
            delay=self.delay,
            driver_indices=self.driver_indices,
            extra_conditioning=self.extra_conditioning,
        )
        restricted = np.hstack([parts["y_past"], parts["faes_current"]])
        full = np.hstack([parts["y_past"], parts["x_past"], parts["faes_current"]])
        self.hy_y_ = _kernel_conditional_entropy(
            np.hstack([parts["y_present"], restricted]),
            restricted,
            radius=self.r,
            metric=self.metric,
            trial_ids=parts["trial_ids"],
        )
        self.conditional_entropy_ = _kernel_conditional_entropy(
            np.hstack([parts["y_present"], full]),
            full,
            radius=self.r,
            metric=self.metric,
            trial_ids=parts["trial_ids"],
        )
        self.transfer_entropy_ = float(self.hy_y_ - self.conditional_entropy_)
        return self


class KernelPartialTransferEntropy(_KernelTEBase):
    """Kernel partial transfer entropy estimator."""

    score_attr_ = "transfer_entropy_"

    def __init__(
        self,
        driver_indices,
        target_indices,
        conditioning_indices,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        r: float = 0.5,
        metric: str = "chebyshev",
        extra_conditioning: str | None = None,
    ):
        super().__init__(
            lags=lags,
            tau=tau,
            delay=delay,
            r=r,
            metric=metric,
            extra_conditioning=extra_conditioning,
        )
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.conditioning_indices = conditioning_indices

    def fit(self, X, y=None):
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
            delay=self.delay,
            driver_indices=self.driver_indices,
            conditioning_indices=self.conditioning_indices,
            extra_conditioning=self.extra_conditioning,
        )
        restricted = np.hstack(
            [parts["y_past"], parts["z_past"], parts["faes_current"]]
        )
        full = np.hstack(
            [parts["y_past"], parts["x_past"], parts["z_past"], parts["faes_current"]]
        )
        self.hy_yz_ = _kernel_conditional_entropy(
            np.hstack([parts["y_present"], restricted]),
            restricted,
            radius=self.r,
            metric=self.metric,
            trial_ids=parts["trial_ids"],
        )
        self.conditional_entropy_ = _kernel_conditional_entropy(
            np.hstack([parts["y_present"], full]),
            full,
            radius=self.r,
            metric=self.metric,
            trial_ids=parts["trial_ids"],
        )
        self.transfer_entropy_ = float(self.hy_yz_ - self.conditional_entropy_)
        return self


class KernelSelfEntropy(_KernelTEBase):
    """Kernel information storage / self-entropy estimator."""

    score_attr_ = "self_entropy_"

    def __init__(
        self,
        target_indices,
        lags: int = 1,
        tau: int = 1,
        r: float = 0.5,
        metric: str = "chebyshev",
    ):
        super().__init__(lags=lags, tau=tau, delay=1, r=r, metric=metric)
        self.target_indices = target_indices

    def fit(self, X, y=None):
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
        )
        self.hy_ = _kernel_entropy(
            parts["y_present"], radius=self.r, metric=self.metric
        )
        self.conditional_entropy_ = _kernel_conditional_entropy(
            np.hstack([parts["y_present"], parts["y_past"]]),
            parts["y_past"],
            radius=self.r,
            metric=self.metric,
            trial_ids=parts["trial_ids"],
        )
        self.self_entropy_ = float(self.hy_ - self.conditional_entropy_)
        return self


class _GaussianTEBase(_TimeSeriesEstimator):
    def __init__(
        self,
        *,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        extra_conditioning: str | None = None,
    ):
        self.lags = lags
        self.tau = tau
        self.delay = delay
        self.extra_conditioning = extra_conditioning

    @staticmethod
    def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == 0:
            return y.copy()
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ coef

    @staticmethod
    def _gaussian_entropy_from_residuals(residuals: np.ndarray) -> float:
        cov = np.atleast_2d(np.cov(residuals, rowvar=False))
        d = residuals.shape[1] if residuals.ndim > 1 else 1
        _, logdet = np.linalg.slogdet(cov)
        return float(0.5 * logdet + 0.5 * d * np.log(2 * np.pi * np.e))

    @staticmethod
    def _compute_f_test(
        res_unrestricted: np.ndarray,
        res_restricted: np.ndarray,
        p_unrestricted: int,
        p_restricted: int,
    ) -> float:
        rss_u = float(np.sum(res_unrestricted**2))
        rss_r = float(np.sum(res_restricted**2))
        num_restrictions = p_unrestricted - p_restricted
        df2 = res_restricted.shape[0] - p_unrestricted
        if num_restrictions <= 0 or df2 <= 0 or rss_u <= 0:
            return 1.0
        f_stat = ((rss_r - rss_u) / num_restrictions) / (rss_u / df2)
        return float(f.sf(f_stat, num_restrictions, df2))


class GaussianTransferEntropy(_GaussianTEBase):
    """Linear-Gaussian bivariate transfer entropy estimator."""

    score_attr_ = "transfer_entropy_"

    def __init__(
        self,
        driver_indices,
        target_indices,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        extra_conditioning: str | None = None,
    ):
        super().__init__(
            lags=lags, tau=tau, delay=delay, extra_conditioning=extra_conditioning
        )
        self.driver_indices = driver_indices
        self.target_indices = target_indices

    def fit(self, X, y=None):
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
            delay=self.delay,
            driver_indices=self.driver_indices,
            extra_conditioning=self.extra_conditioning,
        )
        restricted = np.hstack([parts["y_past"], parts["faes_current"]])
        full = np.hstack([parts["y_past"], parts["x_past"], parts["faes_current"]])
        res_restricted = self._ols_residuals(parts["y_present"], restricted)
        res_unrestricted = self._ols_residuals(parts["y_present"], full)
        self.hy_y_ = self._gaussian_entropy_from_residuals(res_restricted)
        self.conditional_entropy_ = self._gaussian_entropy_from_residuals(
            res_unrestricted
        )
        self.transfer_entropy_ = float(self.hy_y_ - self.conditional_entropy_)
        self.p_value_ = self._compute_f_test(
            res_unrestricted,
            res_restricted,
            p_unrestricted=full.shape[1],
            p_restricted=restricted.shape[1],
        )
        return self


class GaussianPartialTransferEntropy(_GaussianTEBase):
    """Linear-Gaussian partial transfer entropy estimator."""

    score_attr_ = "transfer_entropy_"

    def __init__(
        self,
        driver_indices,
        target_indices,
        conditioning_indices,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        extra_conditioning: str | None = None,
    ):
        super().__init__(
            lags=lags, tau=tau, delay=delay, extra_conditioning=extra_conditioning
        )
        self.driver_indices = driver_indices
        self.target_indices = target_indices
        self.conditioning_indices = conditioning_indices

    def fit(self, X, y=None):
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
            delay=self.delay,
            driver_indices=self.driver_indices,
            conditioning_indices=self.conditioning_indices,
            extra_conditioning=self.extra_conditioning,
        )
        restricted = np.hstack(
            [parts["y_past"], parts["z_past"], parts["faes_current"]]
        )
        full = np.hstack(
            [parts["y_past"], parts["x_past"], parts["z_past"], parts["faes_current"]]
        )
        res_restricted = self._ols_residuals(parts["y_present"], restricted)
        res_unrestricted = self._ols_residuals(parts["y_present"], full)
        self.hy_yz_ = self._gaussian_entropy_from_residuals(res_restricted)
        self.conditional_entropy_ = self._gaussian_entropy_from_residuals(
            res_unrestricted
        )
        self.transfer_entropy_ = float(self.hy_yz_ - self.conditional_entropy_)
        self.p_value_ = self._compute_f_test(
            res_unrestricted,
            res_restricted,
            p_unrestricted=full.shape[1],
            p_restricted=restricted.shape[1],
        )
        return self


class GaussianSelfEntropy(_GaussianTEBase):
    """Linear-Gaussian information storage estimator."""

    score_attr_ = "self_entropy_"

    def __init__(self, target_indices, lags: int = 1, tau: int = 1):
        super().__init__(lags=lags, tau=tau, delay=1)
        self.target_indices = target_indices

    def fit(self, X, y=None):
        X = np.asarray(X)
        X = X - np.mean(X, axis=-2, keepdims=True)
        X_trials = as_trial_array(X)
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
        )
        target_series = X_trials[:, :, self.target_indices[0]].reshape(-1, 1)
        self.hy_ = self._gaussian_entropy_from_residuals(target_series)
        res_unrestricted = self._ols_residuals(parts["y_present"], parts["y_past"])
        self.conditional_entropy_ = self._gaussian_entropy_from_residuals(
            res_unrestricted
        )
        self.self_entropy_ = float(self.hy_ - self.conditional_entropy_)
        self.p_value_ = self._compute_f_test(
            res_unrestricted,
            target_series,
            p_unrestricted=parts["y_past"].shape[1],
            p_restricted=0,
        )
        return self


class GaussianCopulaTransferEntropy(_GaussianTEBase):
    """Transfer entropy after a Gaussian-copula marginal transform."""

    score_attr_ = "transfer_entropy_"

    def __init__(
        self,
        driver_indices,
        target_indices,
        lags: int = 1,
        tau: int = 1,
        delay: int = 1,
        extra_conditioning: str | None = None,
    ):
        super().__init__(
            lags=lags, tau=tau, delay=delay, extra_conditioning=extra_conditioning
        )
        self.driver_indices = driver_indices
        self.target_indices = target_indices

    def fit(self, X, y=None):
        parts = self._embedding_parts(
            X,
            target_index=self.target_indices[0],
            lags=self.lags,
            tau=self.tau,
            delay=self.delay,
            driver_indices=self.driver_indices,
            extra_conditioning=self.extra_conditioning,
        )
        y_present = _gaussian_copula_transform(parts["y_present"])
        y_past = _gaussian_copula_transform(parts["y_past"])
        x_past = _gaussian_copula_transform(parts["x_past"])
        faes_current = _gaussian_copula_transform(parts["faes_current"])

        restricted = np.hstack([y_past, faes_current])
        full = np.hstack([y_past, x_past, faes_current])
        res_restricted = self._ols_residuals(y_present, restricted)
        res_unrestricted = self._ols_residuals(y_present, full)
        self.hy_y_ = self._gaussian_entropy_from_residuals(res_restricted)
        self.conditional_entropy_ = self._gaussian_entropy_from_residuals(
            res_unrestricted
        )
        self.transfer_entropy_ = float(self.hy_y_ - self.conditional_entropy_)
        return self


__all__ = [
    "DirectKSGConditionalMutualInformation",
    "GaussianCopulaConditionalMutualInformation",
    "GaussianCopulaMutualInformation",
    "GaussianCopulaTransferEntropy",
    "GaussianPartialTransferEntropy",
    "GaussianSelfEntropy",
    "GaussianTransferEntropy",
    "KSGEntropy",
    "KSGMutualInformation",
    "KSGPartialTransferEntropy",
    "KSGSelfEntropy",
    "KSGTransferEntropy",
    "KernelPartialTransferEntropy",
    "KernelSelfEntropy",
    "KernelTransferEntropy",
    "MVCondEntropy",
    "MVExponentialEntropy",
    "MVKSGCondEntropy",
    "MVKSGCondMutualInformation",
    "MVKSGPartialInformationDecomposition",
    "MVKSGTransferEntropy",
    "MVLNEntropy",
    "MVNEntropy",
    "MVNMutualInformation",
    "MVParetoEntropy",
]
