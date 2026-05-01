"""Tests for FITS utilities and LENSTOOL comparison helpers."""

import numpy as np
import pytest


def test_write_and_read_fits_roundtrip(tmp_path):
    from gravlensai.utils.fits_utils import write_fits, read_fits

    arr = (np.random.randn(32, 32) * 10).astype(np.float32)
    out = tmp_path / 'test.fits'
    write_fits(arr, str(out), header={'OBSERVER': 'gravlens'})

    data, header = read_fits(str(out), extension='SCI')
    assert data.shape == arr.shape
    assert data.dtype == np.float32
    assert isinstance(header, dict)


def test_fits_info_and_extract_subimage(tmp_path):
    from gravlensai.utils.fits_utils import write_fits, fits_info, extract_subimage

    arr = np.ones((128, 128), dtype=np.float32)
    out = tmp_path / 'info.fits'
    write_fits(arr, str(out))

    info = fits_info(str(out))
    assert info['n_extensions'] >= 1
    assert len(info['extensions']) >= 1

    sub = extract_subimage(arr, center_y=64, center_x=64, size=64)
    assert sub is not None and sub.shape == (64, 64)
    assert extract_subimage(arr, center_y=5, center_x=5, size=64) is None


def test_extract_subimage_rejects_nan():
    from gravlensai.utils.fits_utils import extract_subimage

    arr = np.zeros((64, 64), dtype=np.float32)
    arr[32, 32] = np.nan
    assert extract_subimage(arr, center_y=32, center_x=32, size=64) is None


def test_compare_parameters_and_speedup():
    from gravlensai.evaluate.lenstool_compare import compare_parameters, compute_speedup

    truth = np.array([[1.0, 0.1, -0.1, 0.01, -0.01], [1.2, 0.0, 0.2, -0.01, 0.02]], dtype=np.float32)
    pred = truth + 0.05

    comp = compare_parameters(pred, truth)
    assert 'Einstein_radius' in comp
    assert 'rmse' in comp['Einstein_radius']

    speed = compute_speedup(cnn_total_seconds=0.2, n_images=20, lenstool_mode='default')
    assert speed['speedup_factor'] > 1
    assert speed['cnn_ms_per_image'] > 0


def test_load_lenstool_results(tmp_path):
    from gravlensai.evaluate.lenstool_compare import load_lenstool_results

    csv = tmp_path / 'lenstool.csv'
    csv.write_text(
        'id,theta_E,e1,e2,gamma1,gamma2\n'
        '0,1.1,0.1,-0.1,0.01,-0.01\n'
        '1,1.2,0.2,-0.2,0.02,-0.02\n'
    )

    arr = load_lenstool_results(str(csv))
    assert arr.shape == (2, 5)


def test_load_lenstool_results_missing_columns(tmp_path):
    from gravlensai.evaluate.lenstool_compare import load_lenstool_results

    csv = tmp_path / 'bad.csv'
    csv.write_text('id,foo,bar\n0,1,2\n')

    with pytest.raises(ValueError):
        load_lenstool_results(str(csv))
