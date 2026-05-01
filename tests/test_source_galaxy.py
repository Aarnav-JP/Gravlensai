"""Tests for source/lens galaxy construction helpers."""

import galsim


def test_make_sersic_galaxy_returns_gsobject():
    from gravlensai.simulate.source_galaxy import make_sersic_galaxy

    gal = make_sersic_galaxy(
        sersic_n=1.5,
        half_light_radius=0.3,
        flux=1000.0,
        e1=0.1,
        e2=-0.05,
        offset_x=0.02,
        offset_y=-0.01,
    )
    assert isinstance(gal, galsim.GSObject)


def test_make_sersic_clips_large_ellipticity():
    from gravlensai.simulate.source_galaxy import make_sersic_galaxy

    gal = make_sersic_galaxy(
        sersic_n=2.0,
        half_light_radius=0.4,
        flux=1500.0,
        e1=3.0,
        e2=3.0,
    )
    assert isinstance(gal, galsim.GSObject)


def test_make_lens_and_source_galaxy_helpers():
    from gravlensai.simulate.source_galaxy import make_lens_galaxy, make_source_galaxy

    lens = make_lens_galaxy()
    source = make_source_galaxy(offset_x=0.1, offset_y=-0.1)
    assert isinstance(lens, galsim.GSObject)
    assert isinstance(source, galsim.GSObject)
