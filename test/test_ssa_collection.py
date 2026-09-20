import numpy as np
import pandas as pd
import pytest

from py_ssa_lib.ssa_collection import SSA, MSSA
from py_ssa_lib.svd_strategies import FullSVD, RandomizedSVD
from py_ssa_lib.weights_strategies import (
    UniformWeights,
    PrecomputedWeights,
    DistanceWeights,
    ClusterAwareWeights,
)


def test_default_strategies():
    modelSSA = SSA()
    

    modelMSSA = MSSA()
    assert isinstance(modelMSSA.decomposition_strategy, FullSVD)
    assert isinstance(modelMSSA.weighting_strategy, UniformWeights)

    assert isinstance(modelSSA.decomposition_strategy, FullSVD)
    assert isinstance(modelSSA.weighting_strategy, UniformWeights)


def test_window_length_default_and_bounds():
    # Canonical SSA convention: 2 <= L <= floor((N + 1) / 2), equivalently L <= K.
    odd = SSA().fit(np.arange(11.0))
    assert odd.L == 6
    assert odd.K == 6

    even = SSA().fit(np.arange(12.0))
    assert even.L == 6
    assert even.K == 7

    # Upper boundary is valid for odd N.
    boundary = SSA().fit(np.arange(11.0), L=6)
    assert boundary.L == 6

    with pytest.raises(ValueError):
        SSA().fit(np.arange(11.0), L=1)

    with pytest.raises(ValueError):
        SSA().fit(np.arange(11.0), L=7)


def test_input_semantics_and_pandas_metadata():
    s = pd.Series(np.arange(12.0), name="x")
    ssa = SSA().fit(s, L=4)
    assert ssa.X_s.shape == (12, 1)
    assert ssa.series_name == "x"
    assert ssa.time_index.equals(s.index)

    df = pd.DataFrame({"x": np.arange(12.0), "y": np.arange(12.0) ** 2})
    mssa = MSSA().fit(df, L=4)
    assert mssa.X_s.shape == (12, 2)
    assert mssa.series_names == ["x", "y"]
    assert mssa.time_index.equals(df.index)

    with pytest.raises(ValueError):
        SSA().fit(df.to_numpy(), L=4)
    with pytest.raises(ValueError):
        MSSA().fit(np.arange(12.0), L=4)


def test_full_reconstruction_ssa():
    t = np.arange(40)
    x = 2.0 + 0.03 * t + np.sin(2 * np.pi * t / 10)
    model = SSA().fit(x, L=15)
    x_rec = model.reconstruct_components(np.arange(model.d))[:, 0]
    np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-10)


def test_full_reconstruction_weighted_mssa():
    t = np.arange(40)
    X = np.column_stack([
        np.sin(2 * np.pi * t / 10),
        0.5 * np.cos(2 * np.pi * t / 8),
        1.0 + 0.02 * t,
    ])
    weights = np.array([1.0, 0.6, 0.2])
    model = MSSA(weighting_strategy=PrecomputedWeights()).fit(X, L=15, weights=weights)
    X_rec = model.reconstruct_components(np.arange(model.d))
    np.testing.assert_allclose(X_rec, X, rtol=1e-10, atol=1e-10)


def test_uniform_weights_equal_explicit_ones():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    a = MSSA().fit(X, L=10)
    b = MSSA(weighting_strategy=PrecomputedWeights()).fit(
        X, L=10, weights=np.ones(X.shape[1])
    )
    np.testing.assert_allclose(a.X_ss, b.X_ss)
    np.testing.assert_allclose(a.Sigma, b.Sigma)


def test_hankel_weights():
    model = SSA().fit(np.arange(10.0), L=4)
    expected = np.array([1, 2, 3, 4, 4, 4, 4, 3, 2, 1])
    np.testing.assert_array_equal(model.construct_hankel_weights(), expected)


def test_weighted_correlation_properties():
    t = np.arange(48)
    X = np.column_stack([
        np.sin(2 * np.pi * t / 12),
        np.cos(2 * np.pi * t / 12),
        0.02 * t,
    ])
    model = MSSA(weighting_strategy=PrecomputedWeights()).fit(
        X, L=18, weights=np.array([1.0, 0.7, 0.3])
    )
    W = model.compute_weighted_correlation_matrix()
    np.testing.assert_allclose(W, W.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(W), 1.0, atol=1e-12)
    assert np.all((W >= -1e-12) & (W <= 1.0 + 1e-12))


def test_rank_one_exponential_forecast():
    n, steps, growth = 35, 5, 1.03
    x = growth ** np.arange(n)
    truth = growth ** np.arange(n, n + steps)
    model = SSA().fit(x, L=12)

    pred_l = model.L_forecast(steps, [0], return_full=False)[:, 0]
    pred_k = model.K_forecast(steps, [0], return_full=False)[:, 0]

    np.testing.assert_allclose(pred_l, truth, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pred_k, truth, rtol=1e-10, atol=1e-10)


def test_esprit_recovers_sinusoid_frequency():
    n = 72
    omega_true = 2 * np.pi / 12
    x = np.sin(omega_true * np.arange(n))
    model = SSA().fit(x, L=30)
    _, rho, omega = model.estimate_ESPRIT([0, 1])

    np.testing.assert_allclose(np.sort(np.abs(omega)), [omega_true, omega_true], atol=1e-10)
    np.testing.assert_allclose(rho, 1.0, atol=1e-10)


def test_weighting_strategies():
    X = np.zeros((10, 3))
    distances = [0.0, 1.0, 2.0]

    dw = DistanceWeights().compute_weights(X, distances, bandwidth=2.0)
    expected_dw = np.exp(-0.5 * (np.asarray(distances) / 2.0) ** 2)
    np.testing.assert_allclose(dw, expected_dw)

    cw = ClusterAwareWeights().compute_weights(
        X,
        distances=distances,
        focal_cluster=0,
        neighbour_clusters=[0, 1, 0],
        bandwidth=2.0,
        cluster_penalty=0.25,
    )
    np.testing.assert_allclose(cw, expected_dw * np.array([1.0, 0.25, 1.0]))


def test_randomized_svd_reports_retained_energy_below_or_equal_100():
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    model = SSA(
        decomposition_strategy=RandomizedSVD(n_components=5, random_state=0)
    ).fit(x, L=30)
    assert 0 < model.cumsum_contr[-1] <= 100.0 + 1e-10
    assert model.d == 5
