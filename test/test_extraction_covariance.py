from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from scipy import sparse
from scipy.special import ndtr

from pyreduce import cwrappers, numpy_slitdec
from pyreduce.extract import extract
from pyreduce.extraction_uncertainty import slit_response, combine_swath_covariance
from pyreduce.spectra import Spectrum, Spectra
from pyreduce.trace_model import Trace

pytestmark = pytest.mark.unit


def curved_swath(ncols=40, tilt=.12):
    nrows, osample = 7, 3
    centers = 40.23 + np.arange(ncols) * .013
    curve = np.zeros((ncols, 6))
    curve[:, 1] = tilt
    delta = np.zeros(nrows)
    offset = centers.astype(int)
    nslit = osample * (nrows + 1) + 1
    pixel, column, row, weights = numpy_slitdec._geometry(ncols, nrows, centers-offset, offset, nrows//2, osample, curve, np.zeros(nslit))
    slit = np.exp(-.5*((np.arange(nslit)/osample-4)/1.1)**2)
    slit *= osample/slit.sum()
    spectrum = 200 + 50*np.sin(np.arange(ncols)/5)
    image = np.bincount(pixel, weights*spectrum[column]*slit[row], minlength=nrows*ncols).reshape(ncols, nrows).T
    return image, centers, curve, delta, spectrum, slit, osample


@pytest.mark.parametrize("tilt", [0., .12, -.2])
@pytest.mark.parametrize("lambda_sp", [0., .02])
def test_fitted_profile_response_matches_pixel_perturbation(tilt, lambda_sp):
    image, centers, curve, delta, spectrum, slit, osample = curved_swath(tilt=tilt)
    rng = np.random.default_rng(643)
    image += rng.normal(0, .2, image.shape)
    good = np.ones_like(image, dtype=bool)
    good[2, 15] = False
    arguments = (good, centers, curve, delta, slit, osample, .1, lambda_sp)
    fitted, profile, response = slit_response(image, *arguments)
    perturbation = rng.normal(size=image.shape)
    step = 1e-3
    plus, _, _ = slit_response(image+step*perturbation, *arguments)
    minus, _, _ = slit_response(image-step*perturbation, *arguments)
    numerical = (plus-minus)/(2*step)
    np.testing.assert_allclose(response @ perturbation.T.ravel(), numerical, rtol=2e-5, atol=1e-6)
    assert np.all(response[:, 15*image.shape[0]+2] == 0)


def test_covariance_predicts_independent_poisson_read_realizations():
    image, centers, curve, delta, spectrum, slit, osample = curved_swath(ncols=28)
    good = np.ones_like(image, dtype=bool)
    arguments = (good, centers, curve, delta, slit, osample, .02, 0.)
    mean, profile, response = slit_response(image, *arguments)
    background, read_noise = 25., 2.
    variance = image + 2*(background + read_noise**2)
    covariance = (response*variance.T.ravel()) @ response.T
    rng = np.random.default_rng(905)
    samples = []
    for _ in range(240):
        noisy = rng.poisson(image+background) - rng.poisson(background, image.shape) + rng.normal(0, np.sqrt(2)*read_noise, image.shape)
        fitted, _, _ = slit_response(noisy, *arguments)
        samples.append(fitted)
    samples = np.array(samples)
    # Independent realizations test both a native bin and sums of correlated bins.
    weights = np.zeros((3, image.shape[1]))
    weights[0, 12] = 1
    weights[1, 5:13] = 1
    weights[2, 8:23] = 1
    empirical = np.var(samples @ weights.T, axis=0, ddof=1)
    predicted = np.diag(weights @ covariance @ weights.T)
    np.testing.assert_allclose(empirical/predicted, 1, rtol=.22)
    bias = np.mean(samples, axis=0)-mean
    assert np.sqrt(np.mean(bias**2/covariance.diagonal())) < .18


def test_shared_swath_pixels_are_combined_before_variance():
    first = np.array([[1., .2, 0], [0., .4, .7]])
    second = np.array([[.6, .1, 0], [0., .3, 1.]])
    blocks = [(np.array([0, 1]), np.array([0, 1, 2]), first), (np.array([1, 2]), np.array([1, 2, 3]), second)]
    variance = np.array([2., 5., 11., 17.])
    covariance = combine_swath_covariance(blocks, variance, 3).toarray()
    response = np.zeros((3, 4))
    response[:2, :3] += first
    response[1:, 1:] += second
    np.testing.assert_allclose(covariance, (response*variance) @ response.T)


@pytest.mark.parametrize("backend", [cwrappers, numpy_slitdec])
def test_permanent_bad_pixels_cannot_be_revived(backend):
    image, centers, curve, delta, spectrum, slit, osample = curved_swath()
    image += np.random.default_rng(11).normal(0, .2, image.shape)
    good = np.ones(image.shape, dtype=np.uint8)
    good[3, 15] = 0
    result = backend.slitdec(image, np.ones_like(image), good, centers, curve, delta, osample=osample, lambda_sL=.1, maxiter=100, kappa=6)
    assert result["mask"][3, 15] == 0
    assert result["mask"].sum() > image.size*.9


@pytest.mark.parametrize("suffix", [".fits", ".fits.gz"])
def test_public_variance_path_and_covariance_roundtrip(tmp_path, suffix):
    yy, xx = np.indices((20, 100))
    centers = 10.2 + .004*xx
    image = 200*(ndtr((yy+.5-centers)/1.1)-ndtr((yy-.5-centers)/1.1))
    variance = image+30
    trace = Trace(m=51, pos=np.array([.004, 10.2]), column_range=(0, 100), height=7)
    result, = extract(image, [trace], extraction_height=7, variance=variance, osample=3, swath_width=40, lambda_sf=.1, lambda_sp=0, reject_threshold=0, maxiter=100)
    assert result.mask[0] and result.mask[-1]
    assert not result.mask[50]
    assert sparse.isspmatrix_csr(result.covariance)
    np.testing.assert_allclose(result.sig**2, result.covariance.diagonal(), equal_nan=True)
    assert np.any(result.covariance.diagonal(1) != 0)
    result.cont = np.full(len(result.spec), 2.)
    filename = tmp_path/f'covariance{suffix}'
    Spectra(fits.Header({"BUNIT": "electron", "COVUNIT": "electron2"}), [result]).save(filename)
    with fits.open(filename) as hdus:
        assert hdus["SPECTRA"].header["TUNIT1"] == "electron"
        assert hdus["SPECTRA"].header["TUNIT2"] == "electron"
        assert hdus["COV0000"].header["TUNIT1"] == "electron2"
    restored, = Spectra.read(filename, raw=True).data
    np.testing.assert_array_equal(restored.spec, result.spec)
    np.testing.assert_array_equal(restored.covariance.toarray(), result.covariance.toarray())
    normalized, = Spectra.read(filename, continuum_normalization=True).data
    np.testing.assert_allclose(normalized.covariance.toarray(), result.covariance.toarray()/4, equal_nan=True)
    with pytest.raises(ValueError, match="variance must"):
        extract(image, [trace], variance=np.ones((2, 2)))
    variance[10, 40] = np.nan
    with pytest.raises(ValueError, match="variances must"):
        extract(image, [trace], variance=variance)


def test_preset_profile_response_and_masked_spectral_samples():
    image, centers, curve, delta, spectrum, slit, osample = curved_swath(tilt=0)
    good = np.ones_like(image, dtype=bool)
    good[:, 18:21] = False
    arguments = (good, centers, curve, delta, slit, osample, .1, .02)
    fitted, profile, response = slit_response(image, *arguments, preset=True)
    perturbation = np.random.default_rng(31).normal(size=image.shape)
    plus, _, _ = slit_response(image+1e-3*perturbation, *arguments, preset=True)
    minus, _, _ = slit_response(image-1e-3*perturbation, *arguments, preset=True)
    supported = np.ones(image.shape[1], dtype=bool)
    supported[18:21] = False
    np.testing.assert_allclose((response @ perturbation.T.ravel())[supported], ((plus-minus)/2e-3)[supported], rtol=1e-6, atol=1e-7)
    assert np.all(response[~supported] == 0)
    np.testing.assert_allclose(profile, slit)


def test_all_bad_detector_column_is_not_a_measured_flux():
    yy, xx = np.indices((20, 100))
    image = np.ma.array(200*(ndtr((yy+.5-10.2)/1.1)-ndtr((yy-.5-10.2)/1.1)), mask=np.zeros((20, 100), bool))
    image.mask[:, 50] = True
    variance = np.ma.getdata(image)+30
    trace = Trace(m=51, pos=np.array([10.2]), column_range=(0, 100), height=7)
    result, = extract(image, [trace], extraction_height=7, variance=variance, osample=3, swath_width=40, lambda_sf=.1, lambda_sp=.02, reject_threshold=0, maxiter=100)
    assert result.mask[50]
    assert np.isnan(result.covariance[50, 50])
    assert not result.mask[49] and not result.mask[51]


@pytest.mark.parametrize("backend", ["c", "numpy"])
def test_native_iteration_limit_finishes_same_fixed_point(monkeypatch, backend, caplog):
    monkeypatch.setenv("PYREDUCE_EXTRACTION", backend)
    yy, xx = np.indices((24, 80))
    image = 200*(ndtr((yy+.5-12.2)/1.1)-ndtr((yy-.5-12.2)/1.1))
    image += np.random.default_rng(82).normal(0, .2, image.shape)
    trace = Trace(m=9, pos=np.array([0., 12.2]), column_range=(0, 80), height=9)
    settings = dict(extraction_height=9, variance=np.full_like(image, .04), osample=3, swath_width=80, lambda_sf=.1, lambda_sp=0, reject_threshold=6)
    completed, = extract(image, [trace], maxiter=1, **settings)
    converged, = extract(image, [trace], maxiter=100, **settings)
    np.testing.assert_allclose(completed.spec, converged.spec, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(completed.covariance.toarray(), converged.covariance.toarray(), rtol=1e-8, atol=1e-12, equal_nan=True)
    assert "Newton completed the same fit with a stable rejection mask" in caplog.text


def test_unsettled_rejection_mask_cannot_supply_covariance():
    yy, xx = np.indices((24, 80))
    image = 200*(ndtr((yy+.5-12.2)/1.1)-ndtr((yy-.5-12.2)/1.1))
    image += np.random.default_rng(82).normal(0, .2, image.shape)
    trace = Trace(m=9, pos=np.array([0., 12.2]), column_range=(0, 80), height=9)
    # Aggressive clipping leaves threshold crossings after the alternating solve;
    # the fixed-mask derivative is not valid for that unsettled estimator.
    with pytest.raises(RuntimeError, match="rejection mask changed"):
        extract(image, [trace], extraction_height=9, variance=np.full_like(image, .04), osample=3, swath_width=80, lambda_sf=.1, lambda_sp=0, reject_threshold=2, maxiter=1)


def test_refined_rejection_covariance_matches_pixel_perturbation():
    yy, xx = np.indices((24, 80))
    image = 200*(ndtr((yy+.5-12.2)/1.1)-ndtr((yy-.5-12.2)/1.1))
    image += np.random.default_rng(82).normal(0, .2, image.shape)
    # Only this pixel is stochastic for the conditional response check.
    variance = np.zeros_like(image)
    variance[13, 40] = .04

    def fit(offset):
        perturbed = image.copy()
        perturbed[13, 40] += offset
        trace = Trace(m=9, pos=np.array([0., 12.2]), column_range=(0, 80), height=9)
        result, = extract(perturbed, [trace], extraction_height=9, variance=variance, osample=3, swath_width=80, lambda_sf=.1, lambda_sp=0, reject_threshold=2, maxiter=100)
        return result

    result = fit(0)
    step = 1e-4
    plus, minus = fit(step), fit(-step)
    good = ~result.mask
    assert good[40]
    assert result.sig[40] > 0
    derivative = (plus.spec[good]-minus.spec[good])/(2*step)
    covariance = result.covariance.toarray()[np.ix_(good, good)]
    np.testing.assert_allclose(covariance, .04*np.outer(derivative, derivative), rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(result.sig[good]**2, np.diag(covariance))


def test_faint_uvb_profile_refinement_and_pixel_response():
    # A saved 20-day UVB swath where Newton cycled despite native convergence.
    with np.load(Path(__file__).parent/'data/extraction_covariance/uvb20_swath.npz') as archive:
        inputs = dict(archive)
    for name in ('osample', 'lambda_sf', 'lambda_sp', 'preset'):
        inputs[name] = inputs[name].item()
    native = cwrappers.slitdec(inputs['image'], np.ones_like(inputs['image']), inputs['good'].astype(np.uint8), inputs['ycen'], inputs['slitcurve'], inputs['slitdeltas'], osample=inputs['osample'], lambda_sL=inputs['lambda_sf'], lambda_sP=inputs['lambda_sp'], maxiter=4000, kappa=0)
    fitted, profile, response = slit_response(**(inputs | {'slit': native['slitfunction']}))
    np.testing.assert_allclose(profile.sum(), inputs['osample'])
    assert np.all(np.isfinite(response))
    perturbation = np.random.default_rng(764).normal(size=inputs['image'].shape)
    step = 1e-3
    plus, _, _ = slit_response(**(inputs | {'image': inputs['image']+step*perturbation, 'slit': profile}))
    minus, _, _ = slit_response(**(inputs | {'image': inputs['image']-step*perturbation, 'slit': profile}))
    numerical = (plus-minus)/(2*step)
    np.testing.assert_allclose(response @ perturbation.T.ravel(), numerical, rtol=2e-4, atol=2e-6)
    assert np.all(response[:, ~inputs['good'].T.ravel()] == 0)


@pytest.mark.parametrize("preset", [False, True])
def test_deferred_covariance_response_preserves_fitted_spectrum(preset):
    image, centers, curve, delta, spectrum, slit, osample = curved_swath()
    image += np.random.default_rng(11).normal(0, .2, image.shape)
    good = np.ones_like(image, dtype=bool)
    good[3, 15] = False
    arguments = (image, good, centers, curve, delta, slit, osample, .1, .02)
    fitted, profile, response = slit_response(*arguments, preset=preset)
    deferred, deferred_profile, absent = slit_response(*arguments, preset=preset, compute_response=False)
    np.testing.assert_array_equal(deferred, fitted)
    np.testing.assert_array_equal(deferred_profile, profile)
    assert absent is None
    assert np.all(np.isfinite(response))
