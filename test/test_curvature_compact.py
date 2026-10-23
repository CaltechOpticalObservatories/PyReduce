import numpy as np

from pyreduce.slit_curve import Curvature
from pyreduce.trace_model import Trace, load_traces, save_traces


def test_joint_curvature_survives_trace_representation(tmp_path, monkeypatch):
    traces = [Trace(m=20-i, pos=np.array([.001, 25. + 25*i]), column_range=(0, 256), height=12.) for i in range(3)]
    curvature = Curvature(traces, mode="2D", fit_degree=(2, 1), curve_degree=2, curve_height=12, extraction_height=4)
    fitted = np.array([[[.15, .012], [2e-5, -3e-6], [1e-8, 2e-9]], [[.001, -.0001], [1e-7, 2e-8], [2e-10, -1e-11]]])
    peaks = [np.arange(20., 240., 20.) for _ in traces]
    coefficients = [curvature.eval(x, np.full(x.size, i), fitted) for i, x in enumerate(peaks)]
    measurements = peaks, coefficients, [np.arange(-6, 7)]*3, [np.zeros((11, 13))]*3, [np.ones(256)]*3
    monkeypatch.setattr(curvature, "_determine_curvature_all_lines", lambda image: measurements)
    result = curvature.execute(np.zeros((100, 256)), compute_slitdeltas=False)
    x = np.array([0., 37.5, 128., 255.])
    for i, trace in enumerate(traces):
        trace.slit = result["fitted_coeffs"][i]
        expected = curvature.eval(x, np.full(x.size, i), fitted)
        np.testing.assert_allclose(trace.slit_at_x(x)[1:].T, expected, atol=2e-7)
        np.testing.assert_allclose(trace.slit_at_x(np.arange(256)).T, result["coeffs"][i], atol=1e-12)
    filename = tmp_path / "joint_curvature.fits"
    save_traces(filename, traces)
    loaded, _ = load_traces(filename)
    for before, after in zip(traces, loaded, strict=True):
        assert after.m == before.m
        np.testing.assert_array_equal(after.slit_at_x(x), before.slit_at_x(x))
