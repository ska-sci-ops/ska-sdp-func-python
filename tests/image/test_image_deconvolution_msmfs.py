# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=attribute-defined-outside-init, unused-variable
# pylint: disable=too-many-instance-attributes, invalid-envvar-default
# pylint: disable=consider-using-f-string, logging-not-lazy
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module, import-outside-toplevel
"""Unit tests for image deconvolution vis MSMFS


"""
import pytest

pytestmark = pytest.skip(allow_module_level=True)
import logging
import os
import tempfile
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration import (
    create_named_configuration,
    decimate_configuration,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility import create_visibility

from ska_sdp_func_python.image.deconvolution import (
    deconvolve_list,
    restore_list,
)
from ska_sdp_func_python.image.gather_scatter import (
    image_gather_channels,
    image_scatter_channels,
)
from ska_sdp_func_python.imaging.base import create_image_from_visibility
from ska_sdp_func_python.imaging.imaging import (
    invert_visibility,
    predict_visibility,
)
from ska_sdp_func_python.imaging.weighting import (
    taper_visibility_gaussian,
    weight_visibility,
)

# fix the below imports
from src.ska_sdp_func_python import create_pb
from src.ska_sdp_func_python.image.operations import create_image_from_array
from src.ska_sdp_func_python.imaging.primary_beams import create_low_test_beam
from src.ska_sdp_func_python.simulation import create_low_test_image_from_gleam

log = logging.getLogger("func-python-logger")

log.setLevel(logging.INFO)


class TestImageDeconvolutionMSMFS(unittest.TestCase):
    def setUp(self):

        self.persist = os.getenv("FUNC_PYTHON_PERSIST", False)
        self.niter = 1000
        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.lowcore = decimate_configuration(self.lowcore, skip=3)
        self.nchan = 6
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.linspace(0.9e8, 1.1e8, self.nchan)
        self.channel_bandwidth = numpy.array(
            self.nchan * [self.frequency[1] - self.frequency[0]]
        )
        self.phasecentre = SkyCoord(
            ra=+0.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis = create_visibility(
            config=self.lowcore,
            times=self.times,
            frequency=self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
            polarisation_frame=PolarisationFrame("stokesI"),
            zerow=True,
        )
        self.vis["vis"].data *= 0.0

        # Create model
        self.test_model = create_low_test_image_from_gleam(
            npixel=256,
            cellsize=0.001,
            phasecentre=self.vis.phasecentre,
            frequency=self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            flux_limit=1.0,
        )
        beam = create_low_test_beam(self.test_model)
        self.test_model["pixels"].data *= beam["pixels"].data
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                beam.image_acc.export_to_fits(
                    f"{tempdir}/test_deconvolve_mmclean_beam.fits"
                )
                self.test_model.image_acc.export_to_fits(
                    f"{tempdir}/test_deconvolve_mmclean_model.fits"
                )
        self.vis = predict_visibility(self.vis, self.test_model, context="2d")
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
        self.model = create_image_from_visibility(
            self.vis,
            npixel=512,
            cellsize=0.001,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        self.vis = weight_visibility(self.vis, self.model)
        self.vis = taper_visibility_gaussian(self.vis, 0.002)
        self.dirty, sumwt = invert_visibility(
            self.vis, self.model, context="2d"
        )
        self.psf, sumwt = invert_visibility(
            self.vis, self.model, context="2d", dopsf=True
        )
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                self.dirty.image_acc.export_to_fits(
                    f"{tempdir}/test_deconvolve_mmclean-dirty.fits"
                )
                self.psf.image_acc.export_to_fits(
                    f"{tempdir}/test_deconvolve_mmclean-psf.fits"
                )
        self.dirty = image_scatter_channels(self.dirty)
        self.psf = image_scatter_channels(self.psf)
        window = numpy.ones(shape=self.model["pixels"].shape, dtype=bool)
        window[..., 65:192, 65:192] = True
        self.innerquarter = create_image_from_array(
            window,
            self.model.image_acc.wcs,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        self.innerquarter = image_scatter_channels(self.innerquarter)
        self.sensitivity = create_pb(self.model, "LOW")
        self.sensitivity = image_scatter_channels(self.sensitivity)

    def test_deconvolve_mmclean_no_taylor(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=1,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_no_taylor", 12.806085871833158, -0.14297206892008504
        )

    def test_deconvolve_mmclean_no_taylor_edge(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=1,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="no_edge",
            window_edge=32,
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_no_taylor_edge", 12.806085871833158, -0.1429720689200851
        )

    def test_deconvolve_mmclean_no_taylor_noscales(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0],
            threshold=0.01,
            nmoment=1,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_notaylor_noscales",
            12.874215203967717,
            -0.14419436344642067,
        )

    def test_deconvolve_mmclean_linear(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=2,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_linear", 15.207396524333546, -0.14224980487729696
        )

    def test_deconvolve_mmclean_linear_sensitivity(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            sensitivity=self.sensitivity,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=2,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                sensitivity = image_gather_channels(self.sensitivity)
                sensitivity.image_acc.export_to_fits(
                    f"{tempdir}/test_deconvolve_mmclean_linear_sensitivity.fits"
                )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_linear_sensitivity",
            15.207396524333546,
            -0.14224980487729716,
        )

    def test_deconvolve_mmclean_linear_noscales(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0],
            threshold=0.01,
            nmoment=2,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_linear_noscales", 15.554039669750269, -0.14697685168807129
        )

    def test_deconvolve_mmclean_quadratic(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=3,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_quadratic", 15.302992891627193, -0.15373682171426403
        )

    def test_deconvolve_mmclean_quadratic_noscales(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0],
            threshold=0.01,
            nmoment=3,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_quadratic_noscales",
            15.69172353540307,
            -0.1654330930047646,
        )

    def save_and_check_images(self, tag, flux_max=0.0, flux_min=0.0):
        """Save the images with standard names

        :param tag: Informational, unique tag
        :return:
        """
        cmodel = image_gather_channels(self.cmodel)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                comp = image_gather_channels(self.comp)
                comp.image_acc.export_to_fits(
                    f"{tempdir}/test_deconvolve_{tag}_deconvolved.fits",
                )
                residual = image_gather_channels(self.residual)
                residual.image_acc.export_to_fits(
                    f"{tempdir}/test_deconvolve_{tag}_residual.fits",
                )
                cmodel.image_acc.export_to_fits(
                    f"{tempdir}/test_deconvolve_{tag}_restored.fits",
                )
        qa = cmodel.image_acc.qa_image()
        numpy.testing.assert_allclose(
            qa.data["max"], flux_max, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["min"], flux_min, atol=1e-7, err_msg=f"{qa}"
        )

    def test_deconvolve_mmclean_quadratic_psf_support(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=3,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
            psf_support=32,
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_quadratic_psf", 15.322874439605584, -0.23892365313457908
        )


if __name__ == "__main__":
    unittest.main()
