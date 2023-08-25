"""
Unit tests for geometry calculations
"""
import astropy.units as u
import numpy
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from ska_sdp_func_python.util.geometry import (
    calculate_azel,
    calculate_hourangles,
    calculate_parallactic_angles,
    calculate_transit_time,
    utc_to_ms_epoch,
)

LOCATION = EarthLocation(
    lon=116.76444824 * u.deg, lat=-26.824722084 * u.deg, height=300.0
)
UTC_TIME = Time(["2020-01-01T00:00:00"], format="isot", scale="utc")


def test_calculate_azel(phase_centre):
    """Check calculate_azel returns the correct values"""
    utc_times = Time(
        numpy.arange(0.0, 1.0, 0.1) + UTC_TIME.mjd,
        format="mjd",
        scale="utc",
    )
    azel = calculate_azel(LOCATION, utc_times, phase_centre)
    numpy.testing.assert_array_almost_equal(azel[0][0].deg, -113.964241)
    numpy.testing.assert_array_almost_equal(azel[1][0].deg, 57.715754)
    numpy.testing.assert_array_almost_equal(azel[0][-1].deg, -171.470435)
    numpy.testing.assert_array_almost_equal(azel[1][-1].deg, 81.617363)


def test_calculate_hourangles(phase_centre):
    """Check calculate_hourangles returns the correct values"""
    h_angles = calculate_hourangles(
        LOCATION,
        UTC_TIME,
        phase_centre,
    )
    numpy.testing.assert_array_almost_equal(h_angles[0].deg, 36.881315)


def test_calculate_parallactic_angles(phase_centre):
    """Check calculate_parallactic_angles returns the correct values"""
    p_angles = calculate_parallactic_angles(
        LOCATION,
        UTC_TIME,
        phase_centre,
    )
    numpy.testing.assert_array_almost_equal(p_angles[0].deg, 85.756057)


def test_calculate_transit_time(phase_centre):
    """Check calculate_transit_time returns the correct values"""
    transit_time = calculate_transit_time(
        LOCATION,
        UTC_TIME,
        phase_centre,
    )
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895866)


def test_calculate_transit_time_below_horizon():
    """
    Check calculate_transit_times returns correct values
    below the horizon
    """
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=+80.0 * u.deg, frame="icrs", equinox="J2000"
    )
    transit_time = calculate_transit_time(LOCATION, UTC_TIME, phase_centre)
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895804)


def test_utc_to_ms_epoch():
    """Check utc_to_ms_epoch returns the correct values"""
    ms_epoch = utc_to_ms_epoch(UTC_TIME)
    numpy.testing.assert_array_almost_equal(ms_epoch, 5084553600.0)
