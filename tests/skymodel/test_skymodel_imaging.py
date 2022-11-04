# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=attribute-defined-outside-init, unused-variable
# pylint: disable=too-many-instance-attributes, invalid-envvar-default
# pylint: disable=consider-using-f-string, logging-not-lazy
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module, import-outside-toplevel
""" Regression test for skymodel predict and invert functions
"""
import pytest

pytestmark = pytest.skip(allow_module_level=True)
import logging
import os
import sys
import tempfile
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)

from ska_sdp_func_python.skymodel.skymodel_imaging import (
    skymodel_calibrate_invert,
    skymodel_predict_calibrate,
)

# fix the below imports
from src.ska_sdp_func_python import (
    calculate_visibility_parallactic_angles,
    convert_azelvp_to_radec,
    create_low_test_beam,
    create_low_test_skymodel_from_gleam,
    create_named_configuration,
    ingest_unittest_visibility,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestSkyModel(unittest.TestCase):
    def setUp(self):

        self.persist = os.getenv("FUNC_PYTHON_PERSIST", False)

    def tearDown(self):
        pass

    def actualSetUp(self, freqwin=1, dopol=False, zerow=False):

        self.npixel = 512
        self.low = create_named_configuration("LOW", rmax=300.0)
        self.freqwin = freqwin
        self.vis = []
        self.ntimes = 3
        self.cellsize = 0.001
        self.radius = self.npixel * self.cellsize / 2.0
        # Choose the interval so that the maximum change in w is smallish
        integration_time = numpy.pi * (24 / (12 * 60))
        self.times = numpy.linspace(
            -integration_time * (self.ntimes // 2),
            integration_time * (self.ntimes // 2),
            self.ntimes,
        )

        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(
                freqwin * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.frequency = numpy.array([1.0e8])
            self.channelwidth = numpy.array([4e7])

        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")

        else:
            self.vis_pol = PolarisationFrame("stokesI")
            self.image_pol = PolarisationFrame("stokesI")

        self.phasecentre = SkyCoord(
            ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis = ingest_unittest_visibility(
            self.low,
            self.frequency,
            self.channelwidth,
            self.times,
            self.vis_pol,
            self.phasecentre,
            zerow=zerow,
        )

    def test_predict_no_pb(self):
        """Predict with no primary beam"""
        self.actualSetUp()

        self.skymodel = create_low_test_skymodel_from_gleam(
            npixel=self.npixel,
            cellsize=self.cellsize,
            frequency=self.frequency,
            phasecentre=self.phasecentre,
            radius=self.radius,
            polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=0.3,
            flux_threshold=1.0,
            flux_max=5.0,
        )

        assert len(self.skymodel.components) == 11, len(
            self.skymodel.components
        )
        assert (
            numpy.max(numpy.abs(self.skymodel.image["pixels"].data)) > 0.0
        ), "Image is empty"

        skymodel_vis = skymodel_predict_calibrate(
            self.vis, self.skymodel, context="ng"
        )
        qa = skymodel_vis.visibility_acc.qa_visibility()
        numpy.testing.assert_almost_equal(
            qa.data["maxabs"], 60.35140880932053, err_msg=str(qa)
        )

    def test_predict_with_pb(self):
        """Test predict while applying a time-variable primary beam"""
        self.actualSetUp()

        self.skymodel = create_low_test_skymodel_from_gleam(
            npixel=self.npixel,
            cellsize=self.cellsize,
            frequency=self.frequency,
            phasecentre=self.phasecentre,
            radius=self.radius,
            polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=0.3,
            flux_threshold=1.0,
            flux_max=5.0,
        )

        assert len(self.skymodel.components) == 11, len(
            self.skymodel.components
        )
        assert (
            numpy.max(numpy.abs(self.skymodel.image["pixels"].data)) > 0.0
        ), "Image is empty"

        def get_pb(vis, model):
            pb = create_low_test_beam(model)
            pa = numpy.mean(calculate_visibility_parallactic_angles(vis))
            pb = convert_azelvp_to_radec(pb, model, pa)
            return pb

        skymodel_vis = skymodel_predict_calibrate(
            self.vis,
            self.skymodel,
            context="ng",
            get_pb=get_pb,
        )
        qa = skymodel_vis.visibility_acc.qa_visibility()
        numpy.testing.assert_almost_equal(
            qa.data["maxabs"], 32.20530966848842, err_msg=str(qa)
        )

    def test_invert_no_pb(self):
        """Test invert"""
        self.actualSetUp()

        self.skymodel = create_low_test_skymodel_from_gleam(
            npixel=self.npixel,
            cellsize=self.cellsize,
            frequency=self.frequency,
            phasecentre=self.phasecentre,
            radius=self.radius,
            polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=0.3,
            flux_threshold=1.0,
            flux_max=5.0,
        )

        assert len(self.skymodel.components) == 11, len(
            self.skymodel.components
        )
        assert (
            numpy.max(numpy.abs(self.skymodel.image["pixels"].data)) > 0.0
        ), "Image is empty"

        skymodel_vis = skymodel_predict_calibrate(
            self.vis,
            self.skymodel,
            context="ng",
        )
        assert numpy.max(numpy.abs(skymodel_vis.vis)) > 0.0

        dirty, sumwt = skymodel_calibrate_invert(
            skymodel_vis,
            self.skymodel,
            normalise=True,
            flat_sky=False,
        )
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                dirty.image_acc.export_to_fits(
                    f"{tempdir}/test_skymodel_invert_dirty.fits"
                )
        qa = dirty.image_acc.qa_image()

        numpy.testing.assert_allclose(
            qa.data["max"], 4.179714181498791, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["min"], -0.33300435260339034, atol=1e-7, err_msg=f"{qa}"
        )

    def test_invert_with_pb(self):
        """Test invert while applying a time-variable primary beam"""
        self.actualSetUp()

        self.skymodel = create_low_test_skymodel_from_gleam(
            npixel=self.npixel,
            cellsize=self.cellsize,
            frequency=self.frequency,
            phasecentre=self.phasecentre,
            radius=self.radius,
            polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=0.3,
            flux_threshold=1.0,
            flux_max=5.0,
        )

        assert len(self.skymodel.components) == 11, len(
            self.skymodel.components
        )
        assert (
            numpy.max(numpy.abs(self.skymodel.image["pixels"].data)) > 0.0
        ), "Image is empty"

        def get_pb(bvis, model):
            pb = create_low_test_beam(model)
            pa = numpy.mean(calculate_visibility_parallactic_angles(bvis))
            pb = convert_azelvp_to_radec(pb, model, pa)
            return pb

        skymodel_vis = skymodel_predict_calibrate(
            self.vis,
            self.skymodel,
            context="ng",
            get_pb=get_pb,
        )
        assert numpy.max(numpy.abs(skymodel_vis.vis)) > 0.0

        skymodel = skymodel_calibrate_invert(
            skymodel_vis,
            self.skymodel,
            get_pb=get_pb,
            normalise=True,
            flat_sky=False,
        )
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                skymodel[0].image_acc.export_to_fits(
                    f"{tempdir}/test_skymodel_invert_flat_noise_dirty.fits"
                )
                skymodel[1].image_acc.export_to_fits(
                    f"{tempdir}/test_skymodel_invert_flat_noise_sensitivity.fits"
                )
        qa = skymodel[0].image_acc.qa_image()

        numpy.testing.assert_allclose(
            qa.data["max"], 3.767454977596991, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["min"], -0.23958139130004705, atol=1e-7, err_msg=f"{qa}"
        )

        # Now repeat with flat_sky=True
        skymodel = skymodel_calibrate_invert(
            skymodel_vis,
            self.skymodel,
            get_pb=get_pb,
            normalise=True,
            flat_sky=True,
        )
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                skymodel[0].image_acc.export_to_fits(
                    f"{tempdir}/test_skymodel_invert_flat_sky_dirty.fits"
                )
                skymodel[1].image_acc.export_to_fits(
                    f"{tempdir}/test_skymodel_invert_flat_sky_sensitivity.fits"
                )
        qa = skymodel[0].image_acc.qa_image()

        numpy.testing.assert_allclose(
            qa.data["max"], 4.025153684707801, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["min"], -0.24826345131847594, atol=1e-7, err_msg=f"{qa}"
        )

    def test_predict_nocomponents(self):
        """Test predict with no components"""

        self.actualSetUp()

        self.skymodel = create_low_test_skymodel_from_gleam(
            npixel=self.npixel,
            cellsize=self.cellsize,
            frequency=self.frequency,
            radius=self.radius,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=0.3,
            flux_threshold=1.0,
            flux_max=5.0,
        )

        self.skymodel.components = []

        assert (
            numpy.max(numpy.abs(self.skymodel.image["pixels"].data)) > 0.0
        ), "Image is empty"

        skymodel_vis = skymodel_predict_calibrate(
            self.vis, self.skymodel, context="ng"
        )
        qa = skymodel_vis.visibility_acc.qa_visibility()
        numpy.testing.assert_almost_equal(
            qa.data["maxabs"], 39.916746503252156, err_msg=str(qa)
        )

    def test_predict_noimage(self):
        """Test predict with no image"""

        self.actualSetUp()

        self.skymodel = create_low_test_skymodel_from_gleam(
            npixel=self.npixel,
            cellsize=self.cellsize,
            frequency=self.frequency,
            radius=self.radius,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=0.3,
            flux_threshold=1.0,
            flux_max=5.0,
        )

        self.skymodel.image = None

        assert len(self.skymodel.components) == 11, len(
            self.skymodel.components
        )

        skymodel_vis = skymodel_predict_calibrate(
            self.vis, self.skymodel, context="ng"
        )
        qa = skymodel_vis.visibility_acc.qa_visibility()
        numpy.testing.assert_almost_equal(
            qa.data["maxabs"], 20.434662306068372, err_msg=str(qa)
        )


if __name__ == "__main__":
    unittest.main()
