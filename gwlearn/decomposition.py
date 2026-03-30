"""Prototype of Geographically Weighted Matrix Decomposition — GWPCA, RobustGWPCA."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from libpysal import graph

from .base import BaseDecomposition

__all__ = ["GWPCA", "RobustGWPCA"]


class GWPCA(BaseDecomposition):
    """Geographically Weighted Principal Components Analysis.

    Fits a local PCA at each spatial location via a kernel-weighted covariance
    matrix. Produces a surface of local eigenvectors, eigenvalues, and scores.
    Follows Harris, Brunsdon & Charlton (2011).

    Parameters
    ----------
    n_components : int | None
        Components to retain per location. None keeps all.
    bandwidth : float | int | None
        KNN count (fixed=False) or distance threshold (fixed=True).
    fixed : bool
        Adaptive KNN (False) or fixed distance (True). Default False.
    kernel : str | Callable
        Weight function. Default ``"bisquare"``.
    include_focal : bool
        Include the focal point in its own neighbourhood. Default True.
    graph : libpysal.graph.Graph | None
        Pre-built spatial weights. Overrides bandwidth/kernel if given.
    n_jobs : int
        Joblib parallelism. -1 uses all CPUs.
    fit_global_model : bool
        Also fit a global sklearn PCA as a baseline.
    sign_convention : {"first_positive", "max_abs", "none"}
        Eigenvector sign rule. ``"first_positive"`` matches GWmodel.

    Attributes
    ----------
    components_ : ndarray (n, p, q)
        Local eigenvectors.
    explained_variance_ratio_ : ndarray (n, q)
        Local fraction of variance per component.
    scores_ : ndarray (n, q)
        Local PC scores for each focal observation.
    local_means_ : ndarray (n, p)
        Geographically weighted mean at each location.
    winning_variable_ : pd.Series
        Variable with the highest PC1 loading at each location.
    condition_number_ : pd.Series
        Local covariance condition number (collinearity diagnostic).
    cv_score_ : float | None
        LOO reconstruction error. Set when ``cv=True`` is passed to ``fit``.

    Examples
    --------
    >>> gdf = gpd.read_file(get_path("geoda.guerry")).set_geometry(lambda g: g.centroid)
    >>> X = gdf[["Crm_prs", "Litercy", "Wealth", "Donatns", "Infants"]]
    >>> X = (X - X.mean()) / X.std()
    >>> model = GWPCA(n_components=3, bandwidth=50).fit(X, geometry=gdf.geometry)
    >>> model.explained_variance_ratio_.shape
    (85, 3)
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        bandwidth: float | None = None,
        fixed: bool = False,
        kernel: Literal[
            "triangular", "parabolic", "bisquare", "tricube", "cosine", "boxcar"
        ]
        | Callable = "bisquare",
        include_focal: bool = True,
        graph: graph.Graph | None = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        verbose: bool = False,
        sign_convention: Literal["first_positive", "max_abs", "none"] = "first_positive",
    ):
        self.n_components = n_components
        self.sign_convention = sign_convention
        super().__init__(
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            verbose=verbose,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: None = None,
        geometry: gpd.GeoSeries | None = None,
        cv: bool = False,
    ) -> "GWPCA":
        """Fit GWPCA at every spatial location.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Standardise before calling.
        y : None
            Ignored. Present for sklearn Pipeline compatibility.
        geometry : gpd.GeoSeries | None
            Point geometry. Required unless ``graph`` was supplied at init.
        cv : bool
            Compute LOO CV reconstruction error (Harris et al. 2011, §4.1).

        Returns
        -------
        self
        """
        super().fit(X, y=None, geometry=geometry)

        self._all_eigenvalues = self._eigenvalues.copy()
        q = self.n_components
        if q is not None:
            self._eigenvalues = self._eigenvalues[:, :q]

        self.cv_score_ = None
        if cv:
            self.cv_score_ = self._compute_cv_score(X)

        return self

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        totals = self._all_eigenvalues.sum(axis=1, keepdims=True)
        return np.where(totals > 0, self._eigenvalues / totals, 0.0)
    
    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name,
        focal_x: np.ndarray,
        model_kwargs: dict,
    ) -> list:
        """Fit one local PCA at focal point ``name``."""
        X_local = data.drop(columns=["_weight"]).values.astype(float)
        wt = data["_weight"].values.astype(float)

        if wt.sum() == 0 or len(wt) < 2:
            p = X_local.shape[1]
            q = self.n_components or p
            nan_vec = np.full(p, np.nan)
            return [
                name,
                np.full((p, q), np.nan),
                np.full(q, np.nan),
                np.full(q, np.nan),
                nan_vec,
            ]

        wt_sum = wt.sum()

        weighted_mean = np.average(X_local, axis=0, weights=wt)
        X_centered = X_local - weighted_mean

        X_scaled = X_centered * np.sqrt(wt[:, np.newaxis])
        cov = (X_scaled.T @ X_scaled) / wt_sum

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        sc = self.sign_convention
        if sc == "first_positive":
            signs = np.sign(eigenvectors[0, :])
            signs[signs == 0] = 1.0
            eigenvectors = eigenvectors * signs
        elif sc == "max_abs":
            dom = np.argmax(np.abs(eigenvectors), axis=0)
            signs = np.sign(eigenvectors[dom, np.arange(eigenvectors.shape[1])])
            eigenvectors = eigenvectors * signs

        q = self.n_components
        if q is not None:
            eigenvectors = eigenvectors[:, :q]

        focal_score = (focal_x - weighted_mean) @ eigenvectors

        return [name, eigenvectors, eigenvalues, focal_score, weighted_mean]

    def _compute_cv_score(self, X: pd.DataFrame) -> float:
        """LOO cross-validation reconstruction error (Harris et al. 2011, §4.1)."""
        if self.graph is not None:
            weights = self.graph
        else:
            weights = self._build_weights()

        adjacency = weights._adjacency
        X_vals = X.values.astype(float)
        q = self.n_components

        def _cv_local(focal_id, focal_x):
            nbr_weights = adjacency.loc[focal_id].copy()
            if focal_id in nbr_weights.index:
                nbr_weights[focal_id] = 0.0

            use_mask = nbr_weights > 0
            if use_mask.sum() < 2:
                return np.nan

            nbr_ids = nbr_weights.index[use_mask]
            wt = nbr_weights[use_mask].values.astype(float)

            loc_positions = [X.index.get_loc(idx) for idx in nbr_ids]
            X_nbr = X_vals[loc_positions]

            wt_sum = wt.sum()
            w_mean = np.average(X_nbr, axis=0, weights=wt)
            X_c = X_nbr - w_mean

            X_sc = X_c * np.sqrt(wt[:, np.newaxis])
            cov = (X_sc.T @ X_sc) / wt_sum

            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            if q is not None:
                eigvecs = eigvecs[:, :q]

            x_i = focal_x - w_mean
            reconstructed = x_i @ eigvecs @ eigvecs.T
            return float(np.sum((x_i - reconstructed) ** 2))

        cv_scores = Parallel(n_jobs=self.n_jobs, temp_folder=self.temp_folder)(
            delayed(_cv_local)(fid, X_vals[X.index.get_loc(fid)])
            for fid in self._names
        )

        valid = [s for s in cv_scores if not np.isnan(s)]
        return float(np.sum(valid)) if valid else np.inf

    def stationarity_test(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries,
        component: int = 0,
        n_permutations: int = 99,
        random_state: int | None = None,
    ) -> dict:
        """Monte Carlo permutation test for eigenvalue nonstationarity (Harris 2011, §4.2).

        Returns dict with keys ``"true_sd"``, ``"permuted_sds"``, ``"p_value"``.
        """
        rng = np.random.default_rng(random_state)

        true_sd = float(np.nanstd(self._eigenvalues[:, component]))

        permuted_sds = []
        for _ in range(n_permutations):
            perm_geom = geometry.iloc[rng.permutation(len(geometry))].set_axis(
                geometry.index
            )
            perm_model = GWPCA(
                n_components=self.n_components,
                bandwidth=self.bandwidth,
                fixed=self.fixed,
                kernel=self.kernel,
                include_focal=self.include_focal,
                n_jobs=self.n_jobs,
                fit_global_model=False,
            ).fit(X, geometry=perm_geom)
            permuted_sds.append(
                float(np.nanstd(perm_model._eigenvalues[:, component]))
            )

        permuted_sds = np.array(permuted_sds)
        p_value = float(np.mean(permuted_sds >= true_sd))

        return {
            "true_sd": true_sd,
            "permuted_sds": permuted_sds,
            "p_value": p_value,
        }


    def identify_collinear_locations(
        self,
        threshold: float = 30.0,
    ) -> pd.DataFrame:
        """Flag locations with condition number above ``threshold`` (Harris 2011, §4.5).

        Returns DataFrame with columns: condition_number, pc1_evr, last_pc_evr, is_collinear.
        """
        if not hasattr(self, "_eigenvalues"):
            raise ValueError("Call fit() before identify_collinear_locations().")

        ev = np.abs(self._eigenvalues)               # (n, q)
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = np.where(
                ev[:, -1] > 0,
                ev[:, 0] / ev[:, -1],
                np.inf,
            )
        total = ev.sum(axis=1)
        pc1_evr = np.where(total > 0, ev[:, 0] / total, np.nan)
        last_evr = np.where(total > 0, ev[:, -1] / total, np.nan)

        return pd.DataFrame(
            {
                "condition_number": cond,
                "pc1_evr": pc1_evr,
                "last_pc_evr": last_evr,
                "is_collinear": cond > threshold,
            },
            index=self._names,
        )


class RobustGWPCA(BaseDecomposition):
    """Robust GWPCA using the Minimum Covariance Determinant estimator.

    Replaces the sample covariance in GWPCA with an MCD robust estimate,
    making local PCA resistant to outlier masking. Follows Harris et al. (2014).
    After fitting, use :meth:`outlier_distances` and :meth:`classify_outliers`
    to assign each location to one of four outlier categories (SD × OD).

    Parameters
    ----------
    n_components : int | None
        Components to retain. None keeps all.
    support_fraction : float
        MCD subset fraction. 0.75 matches GWmodel's covMcd(alpha=3/4).
    bandwidth, fixed, kernel, include_focal, graph, n_jobs,
    fit_global_model, keep_models, temp_folder, batch_size, verbose
        Same as :class:`GWPCA`.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        support_fraction: float = 0.75,
        bandwidth: float | None = None,
        fixed: bool = False,
        kernel: str | Callable = "bisquare",
        include_focal: bool = True,
        graph: graph.Graph | None = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.support_fraction = support_fraction
        super().__init__(
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _fit_local(self, model, data, name, focal_x, model_kwargs) -> list:
        """Fit one robust local PCA using MCD covariance at focal point ``name``."""
        from sklearn.covariance import MinCovDet

        X_local = data.drop(columns=["_weight"]).values.astype(float)
        wt = data["_weight"].values.astype(float)
        p = X_local.shape[1]
        q = self.n_components or p

        if len(wt) < max(p + 1, 5):
            return [
                name,
                np.full((p, q), np.nan),
                np.full(q, np.nan),
                np.full(q, np.nan),
                np.full(p, np.nan),
            ]

        robust_center = _weighted_median(X_local, wt)
        X_centered = X_local - robust_center
        X_weighted = X_centered * wt[:, np.newaxis]

        try:
            mcd = MinCovDet(support_fraction=self.support_fraction)
            mcd.fit(X_weighted)
            robust_cov = mcd.covariance_
        except Exception:
            robust_cov = np.cov(X_weighted.T)
            if robust_cov.ndim == 0:
                robust_cov = np.array([[robust_cov]])

        eigenvalues, eigenvectors = np.linalg.eigh(robust_cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        if self.n_components is not None:
            eigenvalues = eigenvalues[: self.n_components]
            eigenvectors = eigenvectors[:, : self.n_components]

        focal_score = (focal_x - robust_center) @ eigenvectors

        return [name, eigenvectors, eigenvalues, focal_score, robust_center]

    def outlier_distances(
        self, X: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Score Distance (SD) and Orthogonal Distance (OD) per location.

        Harris et al. (2014), Eqs. 3-4. Returns ``(sd, od)`` as pd.Series.
        """
        X_vals = X.values.astype(float)
        sd_list, od_list = [], []
        q = self._scores.shape[1]

        for i in range(len(self._names)):
            t_i = self._scores[i]
            v_k = np.abs(self._eigenvalues[i])
            L_q = self._components[i]
            mu_i = self._local_means[i]

            with np.errstate(divide="ignore", invalid="ignore"):
                sd = np.sqrt(np.nansum(np.where(v_k > 0, (t_i**2) / v_k, 0.0)))

            x_i = X_vals[i]
            reconstructed = mu_i + t_i @ L_q.T
            od = float(np.linalg.norm(x_i - reconstructed))

            sd_list.append(float(sd))
            od_list.append(od)

        return (
            pd.Series(sd_list, index=self._names, name="score_distance"),
            pd.Series(od_list, index=self._names, name="orthogonal_distance"),
        )

    def classify_outliers(
        self,
        X: pd.DataFrame,
        sd_cutoff: float | None = None,
        od_cutoff: float | None = None,
        cutoff_method: Literal["robust_zscore", "chi2"] = "robust_zscore",
    ) -> pd.Series:
        """Classify locations into outlier types 0–3 (Harris et al. 2014, §2.3.3).

        0=regular, 1=good leverage, 2=orthogonal outlier, 3=bad leverage.
        ``cutoff_method="chi2"`` uses Group A chi-squared cuts; default uses
        Group B robust z-score (median + 2.5×MAD).
        """
        from scipy.stats import chi2 as _chi2

        sd, od = self.outlier_distances(X)

        def _robust_cutoff(series: pd.Series) -> float:
            median = series.median()
            mad = (series - median).abs().median()
            return float(median + 2.5 * mad * 1.4826)

        def _chi2_sd_cutoff() -> float:
            q = self._scores.shape[1]
            return float(np.sqrt(_chi2.ppf(0.975, df=q)))

        def _chi2_od_cutoff() -> float:
            return float(np.nanpercentile(od.values, 97.5))

        if cutoff_method == "chi2":
            sd_cut = sd_cutoff if sd_cutoff is not None else _chi2_sd_cutoff()
            od_cut = od_cutoff if od_cutoff is not None else _chi2_od_cutoff()
        else:
            sd_cut = sd_cutoff if sd_cutoff is not None else _robust_cutoff(sd)
            od_cut = od_cutoff if od_cutoff is not None else _robust_cutoff(od)

        categories = np.zeros(len(self._names), dtype=int)
        large_sd = sd.values > sd_cut
        large_od = od.values > od_cut

        categories[large_sd & ~large_od] = 1   # good leverage
        categories[~large_sd & large_od] = 2   # orthogonal outlier
        categories[large_sd & large_od] = 3    # bad leverage

        return pd.Series(
            categories,
            index=self._names,
            name="outlier_type",
        )


def _weighted_median(X: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """Weighted median of each column. Used by RobustGWPCA for robust centering."""
    medians = np.empty(X.shape[1])
    wt_norm = wt / wt.sum()
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        cumw = np.cumsum(wt_norm[order])
        pos = np.argmin(np.abs(cumw - 0.5))
        medians[j] = X[order[pos], j]
    return medians
