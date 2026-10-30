# Optimal-extraction detector covariance

`extract(image, traces, variance=variance, ...)` accepts a detector variance
image in the squared units of `image`. Include source, sky, dark and read
variance, including the coefficients of any subtracted or combined frames.
For independent source exposures A1 and A2 using the same sky B, extracting
`A1 + A2 - 2*B` requires `V_A1 + V_A2 + 4*V_B`. Treating the two subtractions
as statistically independent would underestimate the shared sky contribution.

The supplied variance determines uncertainty, not the flux estimator's weights.
PyReduce v0.9 slit decomposition is an unweighted, regularized least-squares
fit of a spectrum and a normalized, oversampled spatial profile. The compiled
solver's pixel-uncertainty argument does not weight those normal equations.
Without the optional variance image, the previous residual-based uncertainty
path remains available. With it, missing or invalid variances and failed fits
raise; no residual-noise substitute is used.

## What is differentiated

For one swath, let `P` be the spectrum, `L` the profile and `d` the detector
pixels. The geometric design matrices satisfy `model = A(L) P = B(P) L`.
The spectral update solves

```
T_P(L, d) = [A.T A + lambda_sp D.T D]^-1 A.T d
```

The profile update solves the corresponding system with `B`, then normalizes
the profile integral to one. Its penalty is recalculated at every iteration:

```
lambda_L(P) = lambda_sf * trace(B.T B) / nslit
T_L(P, d) = normalize([B.T B + lambda_L(P) D.T D]^-1 B.T d)
```

Here `D` is the first-difference operator, exactly as in the native backend.
Both normal systems retain that backend's small diagonal floor. The covariance
calculation first completes the converged alternating fit with the final
rejection mask fixed. It differentiates the normalized updates, including the
spectrum-dependent penalty and active diagonal floor. With

```
U = dT_P/dL,  V = dT_L/dP,  Q = dT_P/dd,  Z = dT_L/dd,
K = dP/dd = Q + U (I - V U)^-1 (V Q + Z),
C_P = K diag(variance) K.T.
```

This includes the response of the estimated spatial profile. Holding that
profile fixed would omit its contribution. An explicitly supplied preset
profile instead gives `K = Q`, conditional on that calibration profile. The
normal systems use banded solves; the coupled solve is only profile-sized.
The same `I - V U` matrix accelerates the normalized-profile fixed point with
Newton updates. Convergence still requires the ordinary alternating spectral
update to change by at most 1e-10 relative to the peak spectrum. If 40 Newton
updates do not converge, restart from the supplied native profile and continue
up to 10,000 ordinary alternating updates with the same mask, regularization
and spectral tolerance. This handles a faint UVB swath where Newton cycles;
in that swath the native fit stops at 967 iterations, so increasing its limit
does not repair refinement. The saved regression fixture checks the converged
response against independently refitted positive and negative pixel
perturbations. Refinement can move pixels across the native rejection threshold
even after native convergence. Extraction then applies the same clipping rule
and refits until the fit and mask agree; non-convergence still raises.

In the preserved faint BLUE
34-order experiment this reduced full-arm extraction from 92 to 38 seconds,
without changing smoothing or relaxing the fixed-point tolerance.
Responses are constructed a swath at a time. Overlapping swath responses are
combined with their actual extraction weights before propagating common-pixel
variance, including cross terms. The full detector-to-spectrum matrix is not
stored in output products.

`Spectrum.covariance` retains within-order covariance as CSR, and `Spectrum.sig`
is the square root of its diagonal. A sum or rebinning with weights `w` has
variance `w.T @ covariance @ w`; summing marginal variances generally loses
correlations. The FITS representation is described in
[Output formats](output_formats.md).

## Interpretation and checks

This is a local propagation of detector fluctuations through the actual fit.
Trace positions, curvature, calibration corrections and the final rejection
mask are held fixed. Their calibration uncertainty is separate. Permanent bad
pixels stay excluded in the C, NumPy and Numba backends. Spectral values with no
detector support stay masked even if spectral regularization can interpolate
them. Data-dependent clipping near its threshold can change discontinuously;
its selection uncertainty is not contained in a fixed-mask derivative.

Regularization can bias a recovered profile or spectrum. That bias is not
random detector variance and is not removed by a covariance calculation.
`lambda_sf` smooths the spatial profile; `lambda_sp` separately smooths the
astrophysical spectrum. A nonzero spectral penalty therefore needs a science
justification. No new spectral smoothing is introduced by requesting variance.

Tests compare the full response against independent pixel perturbations for
positive, zero and negative line tilt and two spectral penalties. Independent
Poisson/source/sky and read-noise realizations check native-bin and summed-bin
variance with the profile refitted. Further tests cover shared swath pixels,
permanent masks, unsupported samples, preset profiles and exact CSR FITS
roundtrips. These checks should be repeated when extending the estimator to
multiple fitted sources or spatially resolved emission: the fitted parameter
vector and its coupled derivatives would then change.

The decomposition and separate profile/spectrum penalties follow
[Piskunov, Wehrhahn & Marquart (2021), Sections 2.3 and 2.8](https://arxiv.org/abs/2008.05827).
The propagated covariance above extends their residual-based uncertainty
treatment to externally specified detector variance and the fitted profile.


## Physical coordinates of measured slit residuals

Trace format 5 stores both `SLITDELTA` (dispersion displacement) and
`SLITDELTA_Y` (spatial offset from the trace center) as double-precision pixel
values. The curvature fit retains the actual coordinates from its calibration
height. Extraction samples these residuals at the same physical subpixels as
the fitted polynomial, independent of the science or flat extraction aperture.
Outside the calibrated residual grid, the fitted polynomial supplies the
curvature; the empirical residual correction is zero, matching the calibration
interpolation boundary rule.

Old trace products with nonzero residuals and no spatial coordinates must be
recalibrated. Their array length cannot determine the calibration height, so no
height is inferred from the requested extraction aperture. Traces with no
residual correction, or an identically zero correction, remain usable.


## Native iteration limits

The native alternating solve stops when its 99th-percentile spectral change is
below `5e-5 * abs(median(spectrum))`. Near a zero median this can converge very
slowly. A finite native result that reaches only the iteration limit may proceed
to the same-equation Newton refinement already required for its response. Other
native failures still fail immediately. After refinement, a single native solve
with the refined profile reconstructs the detector model. If the original
rejection rule changes the mask, the fixed-point fit is recomputed with that
updated mask. The full pixel response is constructed once the mask is stable.
The same configured `maxiter` bounds these rejection updates; a mask that remains unsettled fails explicitly. Permanent bad pixels
and native edge exclusions stay excluded. Covariance is accepted only when the
refined model leaves the rejection mask unchanged. This consistency check also
applies to normally converged native fits.

An unconstrained fitted profile can contain large positive and negative
components when the source is undetected. Numerical convergence does not turn
that profile into a physical measurement. For iteration-limit completions the
log reports the ratio of absolute to signed profile sum: values far above one
identify strong cancellation. No positivity constraint, different extraction
estimator, or larger native iteration allowance is introduced.
