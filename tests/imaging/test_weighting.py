"""
Unit tests for visibility weighting
"""

import logging

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility import create_visibility

from ska_sdp_func_python.image.deconvolution import fit_psf
from ska_sdp_func_python.imaging.base import create_image_from_visibility
from ska_sdp_func_python.imaging.imaging import invert_visibility
from ska_sdp_func_python.imaging.weighting import (
    taper_visibility_gaussian,
    taper_visibility_tukey,
    weight_visibility,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="input_params")
def weighting_fixture():
    """Fixture for weighting.py unit tests"""

    npixel = 512
    image_pol = PolarisationFrame("stokesI")
    lowcore = create_named_configuration("LOWBD2", rmax=600)
    times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
    frequency = numpy.array([1e8])
    channel_bandwidth = numpy.array([1e7])
    vis_pol = PolarisationFrame("stokesI")
    f = numpy.array([100.0])
    numpy.array([f])

    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )

    componentvis = create_visibility(
        lowcore,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        weight=1.0,
        polarisation_frame=vis_pol,
    )

    componentvis["vis"].data *= 0.0

    # Create model
    model = create_image_from_visibility(
        componentvis,
        npixel=npixel,
        cellsize=0.0005,
        nchan=len(frequency),
        polarisation_frame=image_pol,
    )

    params = {
        "componentvis": componentvis,
        "model": model,
    }
    return params


def test_tapering_gaussian(input_params):
    """Apply a Gaussian taper to the visibility and check to see if
    the PSF size is close
    """
    size_required = 0.020
    input_params["componentvis"] = weight_visibility(
        input_params["componentvis"],
        input_params["model"],
        weighting="uniform",
    )
    input_params["componentvis"] = taper_visibility_gaussian(
        input_params["componentvis"], beam=size_required
    )
    psf, _ = invert_visibility(
        input_params["componentvis"],
        input_params["model"],
        dopsf=True,
        context="2d",
    )
    fit = fit_psf(psf)

    assert (
        numpy.abs(fit["bmaj"] - 1.279952050682638) < 1
    ), f"Fit should be {1.279952050682638}, actually is {fit['bmaj']}"


def test_tapering_tukey(input_params):
    """Apply a Tukey window taper and output the psf and FT of the PSF.
       No quantitative check.

    :return:
    """
    input_params["componentvis"] = weight_visibility(
        input_params["componentvis"],
        input_params["model"],
        weighting="uniform",
    )
    input_params["componentvis"] = taper_visibility_tukey(
        input_params["componentvis"], tukey=0.1
    )
    psf, _ = invert_visibility(
        input_params["componentvis"],
        input_params["model"],
        dopsf=True,
        context="2d",
    )
    fit = fit_psf(psf)
    assert (
        numpy.abs(fit["bmaj"] - 0.14492670913355402) < 1.0
    ), f"Fit should be {0.14492670913355402}, actually is {fit['bmaj']}"