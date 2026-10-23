"""Detector-noise propagation through the normalized slit-decomposition fit."""

import numpy as np
from scipy import sparse
from scipy.linalg import cholesky_banded, cho_solve_banded, solve

from .numpy_slitdec import _geometry


def _normal_factor(design, smoothing):
    normal = (design.T @ design).tocsc()
    size = normal.shape[0]
    difference = sparse.diags([-np.ones(size - 1), np.ones(size - 1)], [0, 1], shape=(size - 1, size), format="csc")
    penalty = (difference.T @ difference).tocsc()
    normal = normal + smoothing * penalty
    diagonal = normal.diagonal()
    largest = np.argmax(diagonal)
    floor = diagonal[largest] * 1e-10
    floored = diagonal < floor
    normal = normal + sparse.diags(np.maximum(diagonal, floor) - diagonal)
    lower = sparse.tril(normal).tocoo()
    width = np.max(lower.row - lower.col)
    bands = np.zeros((width + 1, size))
    bands[lower.row - lower.col, lower.col] = lower.data
    factor = cholesky_banded(bands, lower=True, check_finite=False)
    return factor, penalty, floored, largest


def slit_response(image, good, ycen, slitcurve, slitdeltas, slit, osample, lambda_sf, lambda_sp, preset=False):
    """Return the converged spectrum and its derivative with respect to pixels.

    The derivative includes the fitted spatial profile, its normalization and
    the spectrum-dependent spatial smoothing used by the C implementation.
    Geometry and the final rejection mask are held fixed. Pixel columns of the
    response follow ``image.T.ravel()``; no stochastic variance is inferred from
    the background-subtracted image or its fit residuals.
    """
    nrows, ncols = image.shape
    nslit = len(slit)
    offset = ycen.astype(np.int64)
    fraction = ycen - offset
    if len(slitdeltas) == nrows:
        slitdeltas = np.interp(np.linspace(0, nrows - 1, nslit), np.arange(nrows), slitdeltas)
    pixel, col, row, weight = _geometry(ncols, nrows, fraction, offset, nrows // 2, osample, slitcurve, slitdeltas)
    use = good.T.ravel()[pixel]
    pixel, col, row, weight = pixel[use], col[use], row[use], weight[use]
    data = image.T.ravel()
    slit = slit.copy()
    slit *= osample / slit.sum()
    initial_slit = slit.copy()

    # Solve the same alternating-fit fixed point using its profile-sized
    # Jacobian. This avoids hundreds of slowly contracting scalar iterations
    # in faint swaths; the spectral fixed-point tolerance remains unchanged.
    # Near a weak-profile stationary point Newton can cycle. Restart the
    # original alternating updates in that case, retaining the same equations,
    # rejection mask and convergence test before differentiating the fit.
    newton_iterations = 40
    for iteration in range(newton_iterations + 10000):
        if iteration == newton_iterations:
            slit = initial_slit.copy()
        a = sparse.coo_matrix((weight * slit[row], (pixel, col)), shape=(data.size, ncols)).tocsr()
        p_factor, _, p_floor, p_max = _normal_factor(a, lambda_sp)
        spectrum = cho_solve_banded((p_factor, True), a.T @ data, check_finite=False)
        supported = np.array(a.multiply(a).sum(axis=0)).ravel() > 0
        if preset:
            direct = cho_solve_banded((p_factor, True), a.T.toarray(), check_finite=False)
            direct[~supported] = 0
            return spectrum, slit, direct
        b = sparse.coo_matrix((weight * spectrum[col], (pixel, row)), shape=(data.size, nslit)).tocsr()
        smoothing = lambda_sf * b.multiply(b).sum() / nslit
        l_factor, l_penalty, l_floor, l_max = _normal_factor(b, smoothing)
        unnormalized = cho_solve_banded((l_factor, True), b.T @ data, check_finite=False)
        normalization = osample / unnormalized.sum()
        normalize = normalization * (np.eye(nslit) - np.outer(unnormalized / unnormalized.sum(), np.ones(nslit)))
        fixed_slit = unnormalized * normalization
        next_a = sparse.coo_matrix((weight * fixed_slit[row], (pixel, col)), shape=(data.size, ncols)).tocsr()
        next_factor, _, _, _ = _normal_factor(next_a, lambda_sp)
        next_spectrum = cho_solve_banded((next_factor, True), next_a.T @ data, check_finite=False)
        change = np.max(np.abs(next_spectrum-spectrum))
        if iteration >= newton_iterations and change > 1e-10 * np.max(np.abs(spectrum)):
            slit = fixed_slit
            continue
        residual = data - a @ spectrum
        cross = sparse.coo_matrix((weight * residual[pixel], (col, row)), shape=(ncols, nslit)).toarray()
        p_rhs = cross - (a.T @ b).toarray()
        if np.any(p_floor):
            diagonal_derivative = sparse.coo_matrix((2 * weight * a[pixel, col].A1, (col, row)), shape=(ncols, nslit)).toarray()
            p_rhs[p_floor] -= spectrum[p_floor, None] * (1e-10 * diagonal_derivative[p_max] - diagonal_derivative[p_floor])
        u = cho_solve_banded((p_factor, True), p_rhs, check_finite=False)

        a_raw = sparse.coo_matrix((weight * unnormalized[row], (pixel, col)), shape=(data.size, ncols)).tocsr()
        residual_raw = data - b @ unnormalized
        cross_raw = sparse.coo_matrix((weight * residual_raw[pixel], (row, col)), shape=(nslit, ncols)).toarray()
        diagonal_derivative = sparse.coo_matrix((2 * weight * b[pixel, row].A1, (row, col)), shape=(nslit, ncols)).toarray()
        smoothing_derivative = lambda_sf * diagonal_derivative.sum(axis=0) / nslit
        l_rhs = cross_raw - (b.T @ a_raw).toarray() - np.outer(l_penalty @ unnormalized, smoothing_derivative)
        if np.any(l_floor):
            diagonal_derivative += np.outer(l_penalty.diagonal(), smoothing_derivative)
            l_rhs[l_floor] -= unnormalized[l_floor, None] * (1e-10 * diagonal_derivative[l_max] - diagonal_derivative[l_floor])
        v = normalize @ cho_solve_banded((l_factor, True), l_rhs, check_finite=False)
        coupling = np.eye(nslit) - v @ u
        if change <= 1e-10 * np.max(np.abs(spectrum)):
            direct = cho_solve_banded((p_factor, True), a.T.toarray(), check_finite=False)
            z = normalize @ cho_solve_banded((l_factor, True), b.T.toarray(), check_finite=False)
            response = direct + u @ solve(coupling, v @ direct + z, assume_a="gen", check_finite=False)
            response[~supported] = 0
            if not np.all(np.isfinite(response)):
                raise RuntimeError("Non-finite slit-decomposition response")
            return spectrum, slit, response
        # P is the exact spectral solve at L. Thus d(T_L(T_P(L)))/dL = V U.
        slit += solve(coupling, fixed_slit-slit, assume_a="gen", check_finite=False)
        slit *= osample / slit.sum()
    raise RuntimeError("Slit decomposition did not reach a fixed point for uncertainty propagation")


def combine_swath_covariance(swaths, variance, ncol):
    """Propagate common detector pixels through overlapping swath weights.

    Each swath is ``(output_columns, detector_indices, response)``. Responses
    already include the extraction's linear overlap weights. A detector pixel
    occurring in two swaths contributes to their cross covariance as well as
    their individual variances.
    """
    rows, columns, values = [], [], []
    active = []
    for output, pixels, response in swaths:
        covariance = (response * variance[pixels]) @ response.T
        rr, cc = np.meshgrid(output, output, indexing="ij")
        rows.append(rr.ravel())
        columns.append(cc.ravel())
        values.append(covariance.ravel())
        active = [(old_output, old_pixels, old_response) for old_output, old_pixels, old_response in active if old_pixels.max() >= pixels.min()]
        for old_output, old_pixels, old_response in active:
            common, old, new = np.intersect1d(old_pixels, pixels, return_indices=True)
            if len(common):
                cross = (old_response[:, old] * variance[common]) @ response[:, new].T
                rr, cc = np.meshgrid(old_output, output, indexing="ij")
                rows.extend((rr.ravel(), cc.ravel()))
                columns.extend((cc.ravel(), rr.ravel()))
                values.extend((cross.ravel(), cross.ravel()))
        active.append((output, pixels, response))
    return sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(ncol, ncol)).tocsr()
