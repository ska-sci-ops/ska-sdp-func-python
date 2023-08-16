"""
Unit tests for visibility operations
Currently skipped tests:
remove_continuum_visibility:
This function involves polynomial fitting,
cannot operate on a random data.

"""
import numpy
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.visibility.operations import (
    average_visibility_by_channel,
    concatenate_visibility,
    convert_visibility_stokesI_to_polframe,
    convert_visibility_to_stokes,
    convert_visibility_to_stokesI,
    copy_data_and_shrink_polarizations,
    divide_visibility,
    expand_polarizations,
    integrate_visibility_by_channel,
    subtract_visibility,
)


def test_concatenate_visibility(visibility):
    """Unit test for the concatenate_visibility function"""
    new_times = (numpy.pi / 43200.0) * numpy.arange(300.0, 600.0, 30.0)
    new_vis = create_visibility(
        visibility.configuration,
        new_times,
        visibility.frequency.data,
        channel_bandwidth=visibility.channel_bandwidth.data,
        phasecentre=visibility.phasecentre,
        weight=1.0,
    )
    new_ntimes = new_vis.vis.shape[0]
    original_ntimes = visibility.vis.shape[0]
    result = concatenate_visibility([visibility, new_vis], dim="time")
    result_ntimes = result.vis.shape[0]
    assert result_ntimes == new_ntimes + original_ntimes

    reverse_result = concatenate_visibility([new_vis, visibility], dim="time")
    assert reverse_result.time.all() == result.time.all()


def test_subtract(visibility):
    """Unit test for the subtract_visibility function"""
    vis1 = visibility.copy(deep=True)
    vis1["vis"].data[...] = 10.0
    vis2 = visibility.copy(deep=True)
    vis2["vis"].data[...] = 1.0

    result = subtract_visibility(vis1, vis2)
    qa = result.visibility_acc.qa_visibility(context="test_qa")
    assert qa.data["maxabs"] == 9.0


def test_divide_visibility(visibility):
    """Unit test for the divide_visibility function with linear polarisation"""
    vis = visibility.copy(deep=True)
    vis["vis"].data[..., :] = [2.0 + 0.0j, 0.0j, 0.0j, 2.0 + 0.0j]
    new_vis = visibility.copy(deep=True)
    new_vis["vis"].data[..., :] = [1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j]

    result = divide_visibility(vis, new_vis)
    assert result.visibility_acc.nvis == vis.visibility_acc.nvis
    assert numpy.max(numpy.abs(result.vis)) == 2.0, numpy.max(
        numpy.abs(result.vis)
    )


def test_divide_visibility_singular(visibility):
    """Unit test for the divide_visibility function with linear polarisation"""
    vis = visibility.copy(deep=True)
    vis["vis"].data[..., :] = [
        2.0 + 0.0j,
        2.0 + 0.0j,
        2.0 + 0.0j,
        2.0 + 0.0j,
    ]
    new_vis = visibility.copy(deep=True)
    new_vis["vis"].data[..., :] = [
        1.0 + 0.0j,
        1.0 + 0.0j,
        1.0 + 0.0j,
        1.0 + 0.0j,
    ]

    result = divide_visibility(vis, new_vis)
    assert result.visibility_acc.nvis == vis.visibility_acc.nvis
    assert numpy.max(numpy.abs(result.vis)) == 2.0, numpy.max(
        numpy.abs(result.vis)
    )


def test_integrate_visibility_by_channel(visibility):
    """
    Unit test for integrate_visibility_by_channel
    """
    vis = visibility.copy(deep=True)
    new_vis = integrate_visibility_by_channel(vis)
    assert new_vis.vis.shape == (
        vis.vis.shape[0],
        vis.vis.shape[1],
        1,
        vis.vis.shape[3],
    )
    assert new_vis.frequency[0] == numpy.median(vis.frequency.data)
    assert new_vis.channel_bandwidth == numpy.sum(vis.channel_bandwidth)


def test_average_visibility_by_channel(visibility):
    """
    Unit test for average_visibility_by_channel
    """
    vis = visibility.copy(deep=True)
    vis["vis"].data[..., :] = [2.0 + 0.0j, 0.0j, 0.0j, 2.0 + 0.0j]
    new_vis = average_visibility_by_channel(vis, channel_average=2)
    assert len(new_vis) == 3
    assert new_vis[0].vis.shape == (
        vis.vis.shape[0],
        vis.vis.shape[1],
        1,
        vis.vis.shape[3],
    )
    assert numpy.max(numpy.abs(new_vis[0].vis)) == 2.0


def test_convert_visibility_pol(visibility):
    """
    Unit test for convert_visibility_to_stokes,
    convert_visibility_to_stokesI,
    and convert_visibility_stokesI_to_polframe
    """
    vis = visibility.copy(deep=True)
    vis_stokes = convert_visibility_to_stokes(vis)
    assert vis_stokes.visibility_acc.polarisation_frame == PolarisationFrame(
        "stokesIQUV"
    )
    assert vis_stokes.vis.shape == vis.vis.shape

    # convert_to_stokesI doesn't support stokesIQUV
    vis = visibility.copy(deep=True)
    vis_stokesI = convert_visibility_to_stokesI(vis)
    assert vis_stokesI.visibility_acc.polarisation_frame == PolarisationFrame(
        "stokesI"
    )
    assert vis_stokesI.vis.shape == (
        vis.vis.shape[0],
        vis.vis.shape[1],
        vis.vis.shape[2],
        1,
    )

    vis_polframe = convert_visibility_stokesI_to_polframe(
        vis_stokesI, poldef=PolarisationFrame("stokesIQUV")
    )
    assert vis_polframe.visibility_acc.polarisation_frame == PolarisationFrame(
        "stokesIQUV"
    )
    assert vis_polframe.vis.shape == vis_stokes.vis.shape


def test_expand_polarizations():
    """
    Check that visibilities are correclty expanded to 4 polarizations
    """
    n_channels = 4
    n_baselines = 10

    for n_polarizations in numpy.array([1, 2, 4]):
        data_in = numpy.zeros([n_channels, n_baselines, n_polarizations])
        data_expanded = expand_polarizations(data_in)
        assert data_expanded.shape == (n_channels, n_baselines, 4)


def test_shrink_polarizations():
    """
    Check that visibilities are correclty shrunk from 4 polarizations to
    [1, 2, 4] polarizations
    """
    n_channels = 4
    n_baselines = 10

    for n_polarizations in numpy.array([1, 2, 4]):
        data_in = numpy.zeros([n_channels, n_baselines, 4])
        data_shrunk = copy_data_and_shrink_polarizations(
            data_in, n_polarizations
        )
        assert data_shrunk.shape == (n_channels, n_baselines, n_polarizations)
