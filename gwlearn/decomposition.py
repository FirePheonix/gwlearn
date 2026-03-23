"""
Geographically Weighted Matrix Decomposition methods.

Implements unsupervised decomposition algorithms that fit one local model per
spatial observation using geographically weighted (kernel-smoothed) covariance
matrices, following the design principles of gwlearn's supervised estimators.

Classes
-------
GWPCA
    Geographically Weighted Principal Components Analysis.
    Harris, Brunsdon & Charlton (2011), IJGIS 25:1717-1736.

RobustGWPCA
    Robust GWPCA using the Minimum Covariance Determinant estimator.
    Harris et al. (2014), Mathematical Geosciences 46:1-31.

References
----------
Harris P, Brunsdon C, Charlton M (2011). Geographically weighted principal
components analysis. International Journal of Geographical Information Science,
25(11), 1717-1736.

Harris P, Brunsdon C, Charlton M, Juggins S, Clarke A (2014). Multivariate
spatial outlier detection using robust geographically weighted methods.
Mathematical Geosciences, 46(1), 1-31.

Fotheringham AS, Brunsdon C, Charlton ME (2002). Geographically Weighted
Regression: the analysis of spatially varying relationships. Wiley.
"""

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

__all__ = ["GWPCA", "RobustGWPCA", "GWFA"]


class GWPCA(BaseDecomposition):
    """Geographically Weighted Principal Components Analysis (GWPCA).

    Fits a local PCA at each spatial location using a geographically weighted
    covariance matrix. At each focal point *i*, the local covariance is:

        Σ(uᵢ, vᵢ) = Xᵀ W(uᵢ, vᵢ) X

    where W is a diagonal matrix of geographic kernel weights. The eigen-
    decomposition of Σ(uᵢ, vᵢ) yields location-specific principal components,
    eigenvalues, and component scores, forming a *surface* of local PCA results
    rather than a single global model.

    This is the Python equivalent of ``gwpca()`` in the GWmodel R package
    (Harris et al. 2011). The core weighted PCA routine translates GWmodel's
    ``wpca`` function exactly::

        # R (GWmodel):
        wpca <- function(x, wt, ...) {
            local.center <- function(x, wt)
                sweep(x, 2, colSums(sweep(x, 1, wt, '*')) / sum(wt))
            svd(sweep(local.center(x, wt), 1, sqrt(wt), '*'), ...)
        }

    Parameters
    ----------
    n_components : int | None
        Number of principal components to retain at each location.
        If ``None`` all components are kept.
    bandwidth : float | int | None
        Bandwidth. Distance threshold when ``fixed=True``; number of nearest
        neighbours when ``fixed=False``.
    fixed : bool
        Fixed distance (True) or adaptive KNN (False). By default False.
    kernel : str | Callable
        Kernel function. By default ``"bisquare"`` (Harris et al. 2011, Eq. 5).
    include_focal : bool
        Include the focal observation in its own neighbourhood weights.
        GWmodel always includes the focal point so this defaults to ``True``.
    graph : libpysal.graph.Graph | None
        Pre-computed spatial weights. Overrides ``bandwidth``/``kernel`` if given.
    n_jobs : int
        Parallelism for joblib. ``-1`` uses all CPUs.
    fit_global_model : bool
        Fit a global sklearn PCA as a non-GW baseline (``self.global_model``).
    keep_models : bool | str | Path
        Store local eigenvectors. Not required for basic output attributes.
    temp_folder : str | None
        joblib memmapping folder.
    batch_size : int | None
        Fit in batches (reduces peak memory for large datasets).
    verbose : bool
        Print progress.

    Attributes
    ----------
    components_ : np.ndarray, shape (n_locations, n_features, n_components)
        Local eigenvectors (loadings) at each focal point.
    explained_variance_ : np.ndarray, shape (n_locations, n_components)
        Local eigenvalues (variance per component).
    explained_variance_ratio_ : np.ndarray, shape (n_locations, n_components)
        Fraction of total local variance per component.
    scores_ : np.ndarray, shape (n_locations, n_components)
        Local component scores for each focal observation.
    local_means_ : np.ndarray, shape (n_locations, n_features)
        Geographically weighted mean vector at each focal location.
    winning_variable_ : pd.Series
        Feature with the highest absolute loading on PC1 at each location.
    condition_number_ : pd.Series
        Local matrix condition number (diagnostic for GWR collinearity).
    cv_score_ : float or None
        Leave-one-out CV reconstruction error. Computed when ``cv=True``.
    global_model : sklearn.decomposition.PCA
        Global (non-GW) PCA fitted on all data. Only present when
        ``fit_global_model=True``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from gwlearn.decomposition import GWPCA
    >>> gdf = gpd.read_file(get_path("geoda.guerry")).set_geometry(
    ...     lambda g: g.centroid
    ... )
    >>> X = gdf[["Crm_prs", "Litercy", "Wealth", "Donatns", "Infants"]]
    >>> X = (X - X.mean()) / X.std()
    >>> model = GWPCA(n_components=3, bandwidth=50).fit(X, geometry=gdf.geometry)
    >>> model.explained_variance_ratio_.shape
    (85, 3)

    References
    ----------
    Harris P, Brunsdon C, Charlton M (2011). Geographically weighted principal
    components analysis. IJGIS 25:1717-1736.
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
    ):
        self.n_components = n_components
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
            Feature matrix. Harris et al. (2011, §3) recommend standardising to
            zero mean and unit variance before calling.
        y : None
            Ignored (present for sklearn Pipeline compatibility).
        geometry : gpd.GeoSeries | None
            Point geometry for each row. Required unless ``graph`` was supplied.
        cv : bool
            If True, compute the leave-one-out cross-validation reconstruction
            error (``cv_score_``) as described in Harris et al. (2011, §4.1)
            and implemented in GWmodel's ``gwpca.cv``. Adds one full extra pass
            through the data. By default False.

        Returns
        -------
        self
        """
        super().fit(X, y=None, geometry=geometry)

        self.cv_score_ = None
        if cv:
            self.cv_score_ = self._compute_cv_score(X)

        return self

    # ------------------------------------------------------------------
    # Local fit — called by _batch_fit for every focal point
    # ------------------------------------------------------------------

    def _fit_local(
        self,
        model,            # None for GWPCA — kept for interface consistency
        data: pd.DataFrame,
        name,
        focal_x: np.ndarray,
        model_kwargs: dict,
    ) -> list:
        """Fit one local PCA at focal point ``name``.

        Implements the weighted PCA of Harris et al. (2011) exactly as
        GWmodel's ``wpca`` function:

        1. Geographically weighted mean centering
        2. Weighted covariance matrix  Σ = Xᵀ W X / Σwᵢⱼ
        3. Eigendecomposition via ``numpy.linalg.eigh`` (symmetric, stable)
        4. Sort descending, retain top-q components

        Parameters
        ----------
        model : None
            Unused; GWPCA manages its own local routine.
        data : pd.DataFrame
            Rows: neighbourhood observations including focal. Columns include
            original features plus ``"_weight"``.
        name : hashable
            Identifier of the focal point.
        focal_x : np.ndarray, shape (n_features,)
            Feature vector of the focal point.
        model_kwargs : dict
            Unused.

        Returns
        -------
        list : [name, eigenvectors, eigenvalues, focal_score, local_mean]
        """
        X_local = data.drop(columns=["_weight"]).values.astype(float)
        wt = data["_weight"].values.astype(float)

        if wt.sum() == 0 or len(wt) < 2:
            # Degenerate neighbourhood — return NaNs
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

        # Step 1: geographically weighted centering
        # μ̂ = Σ(wⱼ xⱼ) / Σwⱼ   — Harris et al. (2011), §2.3
        weighted_mean = np.average(X_local, axis=0, weights=wt)
        X_centered = X_local - weighted_mean

        # Step 2: weighted covariance
        # Σ(u,v) = Xᵀ W X   — Harris et al. (2011), Eq. 4
        # Equivalent to: X_scaled.T @ X_scaled  where  X_scaled = √w · Xc
        X_scaled = X_centered * np.sqrt(wt[:, np.newaxis])
        cov = (X_scaled.T @ X_scaled) / wt_sum

        # Step 3: eigendecomposition — eigh exploits symmetry & is more stable
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort largest eigenvalue first
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Step 4: retain top-q components
        q = self.n_components
        if q is not None:
            eigenvalues = eigenvalues[:q]
            eigenvectors = eigenvectors[:, :q]

        # Local scores for the focal point only
        focal_score = (focal_x - weighted_mean) @ eigenvectors  # (q,)

        return [name, eigenvectors, eigenvalues, focal_score, weighted_mean]

    # ------------------------------------------------------------------
    # Leave-one-out CV score (Harris et al. 2011, §4.1; GWmodel gwpca.cv)
    # ------------------------------------------------------------------

    def _compute_cv_score(self, X: pd.DataFrame) -> float:
        """Compute the LOO cross-validation reconstruction error.

        Translates GWmodel's ``gwpca.cv`` function:

            For each focal i:
              wt[i] <- 0          # exclude focal from its own calibration
              v <- wpca(x[use,], wt[use], nv=k)$v
              score <- score + sum((x[i,] - x[i,] %*% v %*% t(v))^2)

        Harris et al. (2011, §4.1) recommend minimising this over bandwidths
        for a chosen number of retained components ``q``.

        Parameters
        ----------
        X : pd.DataFrame
            The same feature matrix passed to ``fit``.

        Returns
        -------
        float
            Total LOO reconstruction error across all focal points.
        """
        if self.graph is not None:
            weights = self.graph
        else:
            weights = self._build_weights()

        adjacency = weights._adjacency
        X_vals = X.values.astype(float)
        q = self.n_components

        def _cv_local(focal_id, focal_x):
            # Get neighbourhood weights, zero out the focal observation
            nbr_weights = adjacency.loc[focal_id].copy()
            if focal_id in nbr_weights.index:
                nbr_weights[focal_id] = 0.0

            use_mask = nbr_weights > 0
            if use_mask.sum() < 2:
                return np.nan

            nbr_ids = nbr_weights.index[use_mask]
            wt = nbr_weights[use_mask].values.astype(float)

            # Locate rows in X (handle non-integer index)
            loc_positions = [X.index.get_loc(idx) for idx in nbr_ids]
            X_nbr = X_vals[loc_positions]

            # Weighted mean (without focal)
            wt_sum = wt.sum()
            w_mean = np.average(X_nbr, axis=0, weights=wt)
            X_c = X_nbr - w_mean

            # Weighted covariance
            X_sc = X_c * np.sqrt(wt[:, np.newaxis])
            cov = (X_sc.T @ X_sc) / wt_sum

            # Eigendecompose
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            if q is not None:
                eigvecs = eigvecs[:, :q]

            # Reconstruction error at focal point i
            # score_i = ||x_i - x_i @ V @ Vᵀ||²  (Harris et al. 2011, Eq. 7-8)
            x_i = focal_x - w_mean
            reconstructed = x_i @ eigvecs @ eigvecs.T
            return float(np.sum((x_i - reconstructed) ** 2))

        # Parallel over focal points
        cv_scores = Parallel(n_jobs=self.n_jobs, temp_folder=self.temp_folder)(
            delayed(_cv_local)(fid, X_vals[X.index.get_loc(fid)])
            for fid in self._names
        )

        valid = [s for s in cv_scores if not np.isnan(s)]
        return float(np.sum(valid)) if valid else np.inf

    # ------------------------------------------------------------------
    # Monte Carlo stationarity test  (Harris et al. 2011, §4.2)
    # ------------------------------------------------------------------

    def stationarity_test(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries,
        component: int = 0,
        n_permutations: int = 99,
        random_state: int | None = None,
    ) -> dict:
        """Monte Carlo test for spatial eigenvalue nonstationarity.

        Tests whether the local eigenvalues for a given principal component vary
        significantly across space, as described in Harris et al. (2011, §4.2).

        Procedure
        ---------
        1. Compute the true standard deviation (SD) of the local eigenvalues for
           the chosen component.
        2. For each of ``n_permutations`` iterations, shuffle the observation
           coordinates, refit GWPCA, and record the permuted SD.
        3. The p-value is the fraction of permuted SDs ≥ the true SD.

        This is analogous to the GWR coefficient stationarity test of Brunsdon,
        Fotheringham & Charlton (1998).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same as used for fitting).
        geometry : gpd.GeoSeries
            Point geometry (same as used for fitting).
        component : int
            Which principal component's eigenvalue to test (0-indexed).
        n_permutations : int
            Number of random permutations. By default 99.
        random_state : int | None
            Seed for reproducibility.

        Returns
        -------
        dict with keys:
            ``"true_sd"`` — standard deviation of local eigenvalue (component).
            ``"permuted_sds"`` — array of SDs under the null.
            ``"p_value"`` — fraction of permuted SDs ≥ true SD.
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


class RobustGWPCA(BaseDecomposition):
    """Robust Geographically Weighted PCA for multivariate spatial outlier detection.

    Replaces the standard (sample) covariance in GWPCA with a robust estimate
    from the Minimum Covariance Determinant (MCD) estimator (Rousseeuw 1985),
    making the local PCA resistant to the masking effect of outliers.

    This is the Python translation of GWmodel's ``rwpca`` / ``robustSvd`` functions
    (Harris et al. 2014)::

        # R (GWmodel):
        robustSvd <- function(x) {
            pc <- princomp(covmat=covMcd(x, alpha=3/4)$cov)
            return(list(v=pc$loadings, d=pc$sdev))
        }
        rwpca <- function(x, wt, nu=0, nv=2) {
            mids <- sweep(x, 2, wt.median(x, wt))   # robust centering
            res  <- robustSvd(sweep(mids, 1, wt, '*'))
            res$v <- res$v[, 1:nv]
            return(res)
        }

    After fitting, call :meth:`outlier_distances` to compute the Score Distance
    (SD) and Orthogonal Distance (OD) at each location as defined in
    Harris et al. (2014), Eqs. 3-4.

    Parameters
    ----------
    n_components : int | None
        Number of retained components ``q``. Harris et al. (2014) recommend
        trying multiple values of ``q`` as results vary.
    support_fraction : float
        Fraction of observations used in MCD estimation.
        Set at 0.75 following Varmuza & Filzmoser (2009, p.43) and
        Harris et al. (2014, §2.1): h = ⌊0.75 n⌋.
    bandwidth, fixed, kernel, include_focal, graph, n_jobs,
    fit_global_model, keep_models, temp_folder, batch_size, verbose
        Same as :class:`GWPCA`.

    Attributes
    ----------
    components_, explained_variance_, explained_variance_ratio_,
    scores_, local_means_, winning_variable_, condition_number_
        Same as :class:`GWPCA`.

    References
    ----------
    Harris P et al. (2014). Multivariate spatial outlier detection using robust
    geographically weighted methods. Mathematical Geosciences, 46(1), 1-31.
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
        """Fit robust local PCA using MCD estimator.

        Translates GWmodel's ``rwpca``:
          1. Robust centering via weighted median
          2. Apply kernel weights
          3. MCD robust covariance (sklearn.covariance.MinCovDet)
          4. Eigendecomposition

        Harris et al. (2014, §2.1): support_fraction=0.75 matches covMcd(alpha=3/4).
        """
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

        # Step 1: robust centering via weighted median
        # Translates wt.median() in GWmodel's rwpca
        robust_center = _weighted_median(X_local, wt)
        X_centered = X_local - robust_center

        # Step 2: apply kernel weights (pre-weight for MCD)
        X_weighted = X_centered * wt[:, np.newaxis]

        # Step 3: MCD robust scatter
        # support_fraction=0.75 ≡ covMcd(x, alpha=3/4) in R
        # Harris et al. (2014, §2.1): h = ⌊0.75 n⌋
        try:
            mcd = MinCovDet(support_fraction=self.support_fraction)
            mcd.fit(X_weighted)
            robust_cov = mcd.covariance_
        except Exception:
            # Fall back to sample covariance if MCD fails (too few observations)
            robust_cov = np.cov(X_weighted.T)
            if robust_cov.ndim == 0:
                robust_cov = np.array([[robust_cov]])

        # Step 4: eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(robust_cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        if self.n_components is not None:
            eigenvalues = eigenvalues[: self.n_components]
            eigenvectors = eigenvectors[:, : self.n_components]

        focal_score = (focal_x - robust_center) @ eigenvectors

        return [name, eigenvectors, eigenvalues, focal_score, robust_center]

    # ------------------------------------------------------------------
    # SD / OD outlier distances (Harris et al. 2014, §2.3 / §2.4)
    # ------------------------------------------------------------------

    def outlier_distances(
        self, X: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Compute Score Distance (SD) and Orthogonal Distance (OD) per location.

        These are the two diagnostics used in Harris et al. (2014) to classify
        each spatial observation into one of four outlier categories:

        * **Regular** — small SD, small OD
        * **Good leverage point** — large SD, small OD (stabilises fit)
        * **Orthogonal outlier** — small SD, large OD (harms fit)
        * **Bad leverage point** — large SD, large OD (strongly distorts fit)

        Definitions follow Harris et al. (2014), Eqs. 3-4
        (originally from Hubert et al. 2005):

            SD_i = √(Σ_{k=1}^{q} t²_{ik} / v_k)

            OD_i = ||x_i − μ_i − t_i · Lq^T||

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same observations used in ``fit``).

        Returns
        -------
        sd : pd.Series, shape (n_locations,)
            Score distances indexed by location name.
        od : pd.Series, shape (n_locations,)
            Orthogonal distances indexed by location name.
        """
        X_vals = X.values.astype(float)
        sd_list, od_list = [], []
        q = self._scores.shape[1]

        for i in range(len(self._names)):
            t_i = self._scores[i]                # (q,)
            v_k = np.abs(self._eigenvalues[i])   # (q,)
            L_q = self._components[i]            # (n_features, q)
            mu_i = self._local_means[i]          # (n_features,)

            # Score Distance — Harris et al. (2014), Eq. 3
            with np.errstate(divide="ignore", invalid="ignore"):
                sd = np.sqrt(
                    np.nansum(
                        np.where(v_k > 0, (t_i**2) / v_k, 0.0)
                    )
                )

            # Orthogonal Distance — Harris et al. (2014), Eq. 4
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
    ) -> pd.Series:
        """Classify each observation into an outlier category.

        Uses SD and OD cut-offs to assign each location to one of four types
        defined in Harris et al. (2014, §2.3.3):

            0 — Regular (small SD, small OD)
            1 — Good leverage point (large SD, small OD)
            2 — Orthogonal outlier (small SD, large OD)
            3 — Bad leverage point (large SD, large OD)

        If cut-offs are not supplied, robust z-score cut-offs are used
        (group B in Harris et al. 2014): outlier if robust z-score > 2.5.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        sd_cutoff, od_cutoff : float | None
            Explicit cut-off values. If None, derived from the data.

        Returns
        -------
        pd.Series
            Integer category (0-3) indexed by location name.
        """
        sd, od = self.outlier_distances(X)

        def _robust_cutoff(series: pd.Series) -> float:
            median = series.median()
            mad = (series - median).abs().median()
            return float(median + 2.5 * mad * 1.4826)  # 1.4826 ≈ 1/Φ⁻¹(0.75)

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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class GWFA(BaseDecomposition):
    """Geographically Weighted Factor Analysis (GWFA).

    Fits a local Factor Analysis model at each spatial location using
    geographically weighted kernel smoothing. Unlike GWPCA which maximises
    total explained variance via eigendecomposition, Factor Analysis
    decomposes the local covariance into *shared* (communality) and *unique*
    (specific) variance components:

        X = L · Fᵀ + ε,   ε ~ N(0, Ψ)

    where L is the loadings matrix (n_features × n_factors), F are the
    latent factor scores, and Ψ = diag(ψ₁, …, ψ_p) is the unique (noise)
    variance matrix. The model is fitted via the EM algorithm implemented in
    :class:`sklearn.decomposition.FactorAnalysis`.

    At each focal location *i*:

    1. Geographically weighted centering: μ̂ᵢ = Σ wⱼ xⱼ / Σwⱼ
    2. Weight encoding: X̃ = √w · (X − μ̂)
    3. EM Factor Analysis on X̃ (sklearn, ``max_iter`` / ``tol``)
    4. Report loadings L, factor variances, unique variances Ψ

    This is analogous to applying ``factanal()`` / ``fa()`` locally in R,
    following the GW framework of Fotheringham et al. (2002).

    Parameters
    ----------
    n_components : int
        Number of latent factors to retain. Unlike GWPCA, FA requires
        ``n_components`` to be set explicitly and must be < n_features.
        By default 2.
    max_iter : int
        Maximum EM iterations per local model, by default 1000.
    tol : float
        Convergence tolerance for the EM algorithm, by default 1e-2.
    rotation : str | None
        Factor rotation method passed to :class:`sklearn.decomposition.FactorAnalysis`.
        Supported values: ``"varimax"``, ``"quartimax"``, or ``None`` (no rotation).
        Varimax rotation (Kaiser 1958) maximises loadings' variance to improve
        interpretability. By default ``None``.
    bandwidth, fixed, kernel, include_focal, graph, n_jobs,
    fit_global_model, keep_models, temp_folder, batch_size, verbose
        Same as :class:`GWPCA`.

    Attributes
    ----------
    components_ : np.ndarray, shape (n_locations, n_features, n_components)
        Local factor loadings L at each focal point.
    explained_variance_ : np.ndarray, shape (n_locations, n_components)
        Factor variance (sum of squared loadings per factor) at each location.
    explained_variance_ratio_ : np.ndarray, shape (n_locations, n_components)
        Factor variance as fraction of total communal variance.
    scores_ : np.ndarray, shape (n_locations, n_components)
        Local factor scores for each focal observation.
    local_means_ : np.ndarray, shape (n_locations, n_features)
        Geographically weighted mean vector at each focal location.
    noise_variance_ : np.ndarray, shape (n_locations, n_features)
        Unique (noise) variance Ψ per variable at each location.
        High values indicate variables not well captured by the common factors.
    communalities_ : np.ndarray, shape (n_locations, n_features)
        Communality per variable (sum of squared loadings across factors).
        Complement of ``noise_variance_`` when data are standardised.
    winning_variable_ : pd.Series
        Feature with highest absolute loading on Factor 1 at each location.
    condition_number_ : pd.Series
        Condition number of the local loading matrix (collinearity diagnostic).

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from gwlearn.decomposition import GWFA
    >>> gdf = gpd.read_file(get_path("geoda.guerry"))
    >>> gdf = gdf.set_geometry(gdf.centroid)
    >>> X = gdf[["Crm_prs", "Litercy", "Wealth", "Donatns", "Infants"]]
    >>> X = (X - X.mean()) / X.std()
    >>> model = GWFA(n_components=2, bandwidth=30).fit(X, geometry=gdf.geometry)
    >>> model.communalities_.shape
    (85, 5)

    References
    ----------
    Fotheringham AS, Brunsdon C, Charlton ME (2002). Geographically Weighted
    Regression: the analysis of spatially varying relationships. Wiley.

    Kaiser HF (1958). The varimax criterion for analytic rotation in factor
    analysis. Psychometrika, 23(3), 187-200.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        max_iter: int = 1000,
        tol: float = 1e-2,
        rotation: str | None = None,
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
        self.max_iter = max_iter
        self.tol = tol
        self.rotation = rotation
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
    ) -> "GWFA":
        """Fit GWFA at every spatial location.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Should be standardised (zero mean, unit variance)
            before calling, following Harris et al. (2011, §3).
        y : None
            Ignored (present for sklearn Pipeline compatibility).
        geometry : gpd.GeoSeries | None
            Point geometry for each row. Required unless ``graph`` was supplied.

        Returns
        -------
        self
        """
        self._start = time()
        self.geometry = geometry

        if self.graph is not None:
            weights = self.graph
        else:
            self._validate_geometry(self.geometry)
            weights = self._build_weights()

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Weights built")

        self._setup_model_storage()

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = np.arange(X.shape[1])
        self.n_features_in_ = X.shape[1]

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Fitting {len(X)} local FA models")

        training_output = self._fit_models_batch(X, y=None, weights=weights)

        # GWFA returns 6 values (extra: noise_variance per location)
        names, loadings, factor_vars, scores, means, noise_vars = zip(
            *training_output, strict=False
        )

        self._names = list(names)
        self._components = np.array(loadings)        # (n_loc, n_features, n_components)
        self._eigenvalues = np.array(factor_vars)    # (n_loc, n_components)
        self._scores = np.array(scores)              # (n_loc, n_components)
        self._local_means = np.array(means)          # (n_loc, n_features)
        self._noise_variances = np.array(noise_vars) # (n_loc, n_features)

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Local FA models fitted")

        if self.fit_global_model:
            self._fit_global_model_fa(X)

        return self

    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name,
        focal_x: np.ndarray,
        model_kwargs: dict,
    ) -> list:
        """Fit one local Factor Analysis at focal point ``name``.

        Uses sklearn's EM-based FactorAnalysis on kernel-weighted data:

        1. Geographically weighted centering (same as GWPCA)
        2. Apply √w to encode spatial smoothing
        3. EM Factor Analysis via sklearn
        4. Fall back to eigendecomposition on convergence failure

        Parameters
        ----------
        model : None
            Unused; GWFA manages its own local routine.
        data : pd.DataFrame
            Neighbourhood rows with ``"_weight"`` column.
        name : hashable
            Focal point identifier.
        focal_x : np.ndarray, shape (n_features,)
            Feature vector of the focal point.
        model_kwargs : dict
            Unused.

        Returns
        -------
        list : [name, loadings, factor_variance, focal_score, local_mean, noise_var]
        """
        from sklearn.decomposition import FactorAnalysis

        X_local = data.drop(columns=["_weight"]).values.astype(float)
        wt = data["_weight"].values.astype(float)
        p = X_local.shape[1]
        q = self.n_components

        _nan = [
            name,
            np.full((p, q), np.nan),
            np.full(q, np.nan),
            np.full(q, np.nan),
            np.full(p, np.nan),
            np.full(p, np.nan),
        ]

        if wt.sum() == 0 or len(wt) < max(q + 2, 5):
            return _nan

        # Step 1: geographically weighted centering
        weighted_mean = np.average(X_local, axis=0, weights=wt)
        X_centered = X_local - weighted_mean

        # Step 2: encode kernel weights via √w scaling
        X_scaled = X_centered * np.sqrt(wt[:, np.newaxis])

        try:
            fa = FactorAnalysis(
                n_components=q,
                max_iter=self.max_iter,
                tol=self.tol,
                rotation=self.rotation,
            )
            fa.fit(X_scaled)

            # components_ shape: (n_components, n_features) → transpose to (n_features, n_components)
            loadings = fa.components_.T  # (n_features, n_components)

            # Factor variance: sum of squared loadings per factor column
            factor_variance = np.sum(loadings**2, axis=0)  # (n_components,)

            # Unique (noise) variances Ψ — diagonal of residual covariance
            noise_var = fa.noise_variance_  # (n_features,)

            # Factor scores for the focal observation
            focal_score = fa.transform(
                (focal_x - weighted_mean).reshape(1, -1)
            )[0]

        except Exception:
            # Fallback: eigendecomposition of weighted covariance (= GWPCA step)
            # Triggered when EM fails to converge or neighbourhood is near-singular
            wt_sum = wt.sum()
            cov = (X_scaled.T @ X_scaled) / wt_sum
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = np.argsort(eigenvalues)[::-1]
            loadings = eigenvectors[:, order][:, :q]
            factor_variance = eigenvalues[order][:q]
            focal_score = (focal_x - weighted_mean) @ loadings
            noise_var = np.full(p, np.nan)

        return [name, loadings, factor_variance, focal_score, weighted_mean, noise_var]

    def _fit_global_model_fa(self, X: pd.DataFrame):
        """Fit global FactorAnalysis as non-GW baseline stored in ``self.global_model``."""
        from sklearn.decomposition import FactorAnalysis

        self.global_model = FactorAnalysis(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            rotation=self.rotation,
        )
        self.global_model.fit(X)

    # ------------------------------------------------------------------
    # GWFA-specific properties
    # ------------------------------------------------------------------

    @property
    def noise_variance_(self) -> np.ndarray:
        """Unique (noise) variance per variable at each location.

        Shape: ``(n_locations, n_features)``.
        These are the diagonal elements of Ψ in the factor model X = LFᵀ + ε.
        Variables with high unique variance are poorly explained by the local
        common factors.
        """
        return self._noise_variances

    @property
    def communalities_(self) -> np.ndarray:
        """Communality per variable at each location.

        Shape: ``(n_locations, n_features)``.
        ``communality_j = Σ_k L²_{jk}`` — the fraction of variable *j*'s total
        variance captured by the ``n_components`` common factors.  When the data
        are standardised (unit variance), ``communality_j + noise_variance_j ≈ 1``.
        """
        # components_ is (n_loc, n_features, n_components)
        # sum squared loadings over the components axis
        return np.sum(self._components**2, axis=2)  # (n_loc, n_features)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _weighted_median(X: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """Compute the weighted median of each column.

    Translates GWmodel's ``wt.median`` function used inside ``rwpca``::

        wt.median.1 <- function(x, wt) {
            ox <- order(x)
            wox <- cumsum(wt[ox])
            posn <- which.min(abs(wox - 0.5))
            return(x[ox][posn])
        }

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    wt : np.ndarray, shape (n,)

    Returns
    -------
    np.ndarray, shape (p,)
    """
    medians = np.empty(X.shape[1])
    wt_norm = wt / wt.sum()  # normalise to sum to 1
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        cumw = np.cumsum(wt_norm[order])
        pos = np.argmin(np.abs(cumw - 0.5))
        medians[j] = X[order[pos], j]
    return medians
