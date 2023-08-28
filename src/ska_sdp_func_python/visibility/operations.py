"""
Visibility operations
"""

__all__ = [
    "concatenate_visibility_frequency",
    "concatenate_visibility",
    "subtract_visibility",
    "remove_continuum_visibility",
    "divide_visibility",
    "integrate_visibility_by_channel",
    "average_visibility_by_channel",
    "convert_visibility_to_stokes",
    "convert_visibility_to_stokesI",
    "convert_visibility_stokesI_to_polframe",
    "expand_polarizations",
    "copy_data_and_shrink_polarizations",
]

import logging
from typing import List

import numpy
import xarray
from ska_sdp_datamodels.science_data_model.polarisation_functions import (
    convert_circular_to_stokes,
    convert_circular_to_stokesI,
    convert_linear_to_stokes,
    convert_linear_to_stokesI,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility.vis_model import Visibility

log = logging.getLogger("func-python-logger")


def concatenate_visibility(vis_list, dim="time"):
    """Concatenate a list of visibilities.

    :param vis_list: List of vis
    :param dim: Name of the dimension to concatenate along
    :return: Concatenated visibility
    """
    if not len(vis_list) > 0:
        raise ValueError("concatenate_visibility: vis_list is empty")

    concatenated_vis = xarray.concat(
        vis_list,
        dim=dim,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    try:
        setattr(
            concatenated_vis,
            "_imaging_weight",
            xarray.concat(
                [
                    vis._imaging_weight  # pylint: disable=protected-access
                    for vis in vis_list
                ],
                dim=dim,
            ),
        )
    except TypeError:
        # if vis._imaging_weight is None, concat throws a TypeError
        setattr(concatenated_vis, "_imaging_weight", None)

    return concatenated_vis


def concatenate_visibility_frequency(bvis_list):
    """Concatenate a list of Visibility's in frequency.

    The list should be in sequence of channels.

    :param bvis_list: List of Visibility
    :return: Visibility
    """
    return concatenate_visibility(bvis_list, "frequency")


def subtract_visibility(vis, model_vis, inplace=False):
    """Subtract model_vis from vis, returning new visibility.

    :param vis: Visibility to be subtracted from
    :param model_vis: Visibility to subtract with
    :return: Subtracted visibility
    """

    assert vis.vis.shape == model_vis.vis.shape, (
        f"Observed {vis.vis.shape} and model visibilities "
        f"{model_vis.vis.shape} have different shapes."
    )

    if inplace:
        vis["vis"].data = vis["vis"].data - model_vis["vis"].data
        return vis

    residual_vis = vis.copy(deep=True)
    residual_vis["vis"].data = residual_vis["vis"].data - model_vis["vis"].data
    return residual_vis


def remove_continuum_visibility(
    vis: Visibility, degree=1, mask=None
) -> Visibility:
    """Fit and remove continuum visibility.

    Fit a polynomial in frequency of the specified degree where mask is True.

    :param vis: Visibility
    :param degree: Degree of polynomial
    :param mask: Mask of continuum
    :return: Visibility
    """

    if mask is not None:
        assert numpy.sum(mask) > 2 * degree, "Insufficient channels for fit"

    nchan = len(vis.frequency)
    # This loop needs to be optimised
    x = (vis.frequency - vis.frequency[nchan // 2]) / (
        vis.frequency[0] - vis.frequency[nchan // 2]
    )
    for row in range(vis.visibility_acc.nvis):
        for ibaseline, _ in enumerate(vis.baselines):
            for pol in range(vis.visibility_acc.polarisation_frame.npol):
                wt = numpy.sqrt(
                    vis.visibility_acc.flagged_weight[row, ibaseline, :, pol]
                )
                if mask is not None:
                    wt[mask] = 0.0
                fit = numpy.polyfit(
                    x, vis["vis"][row, ibaseline, :, pol], w=wt, deg=degree
                )
                prediction = numpy.polyval(fit, x)
                vis["vis"][row, ibaseline, :, pol] -= prediction
    return vis


def divide_visibility(vis: Visibility, modelvis: Visibility):
    """
    Divide visibility by model forming
    visibility for equivalent point source.

    This is a useful intermediate product for calibration.
    Variation of the visibility in time and frequency due
    to the model structure is removed and the data can be
    averaged to a limit determined by the instrumental stability.
    The weight is adjusted to compensate for the division.

    Zero divisions are avoided and the corresponding weight set to zero.

    :param vis: Visibility to be divided
    :param modelvis: Visibility to divide with
    :return: Divided Visibility
    """

    x = numpy.zeros_like(vis.visibility_acc.flagged_vis)
    xwt = (
        numpy.abs(modelvis.visibility_acc.flagged_vis) ** 2
        * vis.visibility_acc.flagged_weight
    )
    mask = xwt > 0.0
    x[mask] = (
        vis.visibility_acc.flagged_vis[mask]
        / modelvis.visibility_acc.flagged_vis[mask]
    )

    pointsource_vis = Visibility.constructor(
        flags=vis.flags.data,
        baselines=vis.baselines,
        frequency=vis.frequency.data,
        channel_bandwidth=vis.channel_bandwidth.data,
        phasecentre=vis.phasecentre,
        configuration=vis.configuration,
        uvw=vis.uvw.data,
        time=vis.time.data,
        integration_time=vis.integration_time.data,
        vis=x,
        weight=xwt,
        source=vis.source,
        meta=vis.meta,
        polarisation_frame=vis.visibility_acc.polarisation_frame,
    )
    pointsource_vis.imaging_weight = vis.imaging_weight
    return pointsource_vis


def integrate_visibility_by_channel(vis: Visibility) -> Visibility:
    """Integrate visibility across all channels, returning new visibility.

    :param vis: Visibility
    :return: Visibility
    """
    vis_shape = list(vis.vis.shape)
    nchan = vis_shape[2]
    vis_shape[-2] = 1
    flags = numpy.sum(vis.flags.data, axis=-2)[..., numpy.newaxis, :]
    flags[flags < nchan] = 0
    flags[flags > 1] = 1

    newvis = numpy.sum(
        vis["vis"].data * vis.visibility_acc.flagged_weight, axis=-2
    )[..., numpy.newaxis, :]
    newweights = numpy.sum(vis.visibility_acc.flagged_weight, axis=-2)[
        ..., numpy.newaxis, :
    ]
    newimaging_weights = numpy.sum(
        vis.visibility_acc.flagged_imaging_weight, axis=-2
    )[..., numpy.newaxis, :]
    mask = (1 - flags) * newweights > 0.0
    newvis[mask] = newvis[mask] / ((1 - flags) * newweights)[mask]

    new_vis = Visibility.constructor(
        frequency=numpy.ones([1]) * numpy.average(vis.frequency.data),
        channel_bandwidth=numpy.ones([1])
        * numpy.sum(vis.channel_bandwidth.data),
        baselines=vis.baselines,
        phasecentre=vis.phasecentre,
        configuration=vis.configuration,
        uvw=vis.uvw.data,
        time=vis.time.data,
        vis=newvis,
        flags=flags,
        weight=newweights,
        integration_time=vis.integration_time.data,
        polarisation_frame=vis.visibility_acc.polarisation_frame,
        source=vis.source,
        meta=vis.meta,
    )
    new_vis.imaging_weight = newimaging_weights
    return new_vis


def average_visibility_by_channel(
    vis: Visibility, channel_average=None
) -> List[Visibility]:
    """
    Average visibility by groups of channels,
    returning list of new visibility.

    :param vis: Visibility
    :param channel_average: Number of channels to average
    :return: List[Visibility]
    """
    vis_shape = list(vis.vis.shape)
    nchan = vis_shape[2]

    newvis_list = []
    ochannels = range(nchan)

    channels = []
    for i in range(0, nchan, channel_average):
        channels.append([ochannels[i], ochannels[i + channel_average - 1] + 1])
    for group in channels:
        vis_shape[-2] = 1
        freq = numpy.array([numpy.average(vis.frequency[group[0] : group[1]])])
        cb = numpy.array(
            [numpy.sum(vis.channel_bandwidth[group[0] : group[1]])]
        )
        newvis = Visibility.constructor(
            frequency=freq,
            channel_bandwidth=cb,
            baselines=vis.baselines,
            phasecentre=vis.phasecentre,
            configuration=vis.configuration,
            uvw=vis.uvw,
            time=vis.time,
            vis=numpy.zeros(vis_shape, dtype="complex"),
            flags=numpy.zeros(vis_shape, dtype="int"),
            weight=numpy.zeros(vis_shape, dtype="float"),
            integration_time=vis.integration_time,
            polarisation_frame=vis.visibility_acc.polarisation_frame,
            source=vis.source,
            meta=vis.meta,
        )

        newvis.imaging_weight = numpy.zeros(vis_shape, dtype="float")

        vf = vis.flags[..., group[0] : group[1], :]
        vfvw = (
            vis.visibility_acc.flagged_vis[..., group[0] : group[1], :]
            * vis.weight[..., group[0] : group[1], :]
        )
        vfw = vis.visibility_acc.flagged_weight[..., group[0] : group[1], :]
        vfiw = vis.visibility_acc.flagged_imaging_weight[
            ..., group[0] : group[1], :
        ]

        newvis["flags"].data[..., 0, :] = numpy.sum(vf, axis=-2)
        newvis["flags"].data[newvis["flags"].data < nchan] = 0
        newvis["flags"].data[newvis["flags"].data > 1] = 1

        newvis["vis"].data[..., 0, :] = numpy.sum(vfvw, axis=-2)
        newvis["weight"].data[..., 0, :] = numpy.sum(vfw, axis=-2)
        newvis.imaging_weight[..., 0, :] = numpy.sum(vfiw, axis=-2)
        mask = newvis.visibility_acc.flagged_weight > 0.0
        newvis["vis"].data[mask] = (
            newvis["vis"].data[mask]
            / newvis.visibility_acc.flagged_weight[mask]
        )

        newvis_list.append(newvis)

    return newvis_list


def convert_visibility_to_stokes(vis):
    """Convert the polarisation frame data into Stokes parameters.

    :param vis: Visibility
    :return: Converted visibility data.
    """
    poldef = vis.visibility_acc.polarisation_frame
    if poldef == PolarisationFrame("linear"):
        vis["vis"].data[...] = convert_linear_to_stokes(
            vis["vis"].data, polaxis=3
        )
        vis["flags"].data[...] = numpy.logical_or(
            vis.flags.data[..., 0], vis.flags.data[..., 3]
        )[..., numpy.newaxis]
        vis.attrs["_polarisation_frame"] = "stokesIQUV"

    elif poldef == PolarisationFrame("circular"):
        vis["vis"].data[...] = convert_circular_to_stokes(
            vis["vis"].data, polaxis=3
        )
        vis["flags"].data[...] = numpy.logical_or(
            vis.flags.data[..., 0], vis.flags.data[..., 3]
        )[..., numpy.newaxis]
        vis.attrs["_polarisation_frame"] = "stokesIQUV"
    return vis


def convert_visibility_to_stokesI(vis):
    """Convert the polarisation frame data into Stokes I
    dropping other polarisations, return new Visibility.

    :param vis: visibility
    :return: Converted visibility data.
    """
    if vis.visibility_acc.polarisation_frame == PolarisationFrame("stokesI"):
        return vis

    polarisation_frame = PolarisationFrame("stokesI")
    poldef = vis.visibility_acc.polarisation_frame
    if poldef == PolarisationFrame("linear"):
        vis_data = convert_linear_to_stokesI(vis.visibility_acc.flagged_vis)
        vis_flags = numpy.logical_or(
            vis.flags.data[..., 0], vis.flags.data[..., 3]
        )[..., numpy.newaxis]
        vis_weight = (
            vis.visibility_acc.flagged_weight[..., 0]
            + vis.visibility_acc.flagged_weight[..., 3]
        )[..., numpy.newaxis]
        vis_imaging_weight = (
            vis.visibility_acc.flagged_imaging_weight[..., 0]
            + vis.visibility_acc.flagged_imaging_weight[..., 3]
        )[..., numpy.newaxis]
    elif poldef == PolarisationFrame("linearnp"):
        vis_data = convert_linear_to_stokesI(vis.visibility_acc.flagged_vis)
        vis_flags = numpy.logical_or(
            vis.flags.data[..., 0], vis.flags.data[..., 1]
        )[..., numpy.newaxis]
        vis_weight = (
            vis.visibility_acc.flagged_weight[..., 0]
            + vis.visibility_acc.flagged_weight[..., 1]
        )[..., numpy.newaxis]
        vis_imaging_weight = (
            vis.visibility_acc.flagged_imaging_weight[..., 0]
            + vis.visibility_acc.flagged_imaging_weight[..., 1]
        )[..., numpy.newaxis]
    elif poldef == PolarisationFrame("circular"):
        vis_data = convert_circular_to_stokesI(vis.visibility_acc.flagged_vis)
        vis_flags = numpy.logical_or(
            vis.flags.data[..., 0], vis.flags.data[..., 3]
        )[..., numpy.newaxis]
        vis_weight = (
            vis.visibility_acc.flagged_weight[..., 0]
            + vis.visibility_acc.flagged_weight[..., 3]
        )[..., numpy.newaxis]
        vis_imaging_weight = (
            vis.visibility_acc.flagged_imaging_weight[..., 0]
            + vis.visibility_acc.flagged_imaging_weight[..., 3]
        )[..., numpy.newaxis]
    elif poldef == PolarisationFrame("circularnp"):
        vis_data = convert_circular_to_stokesI(vis.visibility_acc.flagged_vis)
        vis_flags = numpy.logical_or(
            vis.flags.data[..., 0], vis.flags.data[..., 1]
        )[..., numpy.newaxis]
        vis_weight = (
            vis.visibility_acc.flagged_weight[..., 0]
            + vis.visibility_acc.flagged_weight[..., 1]
        )[..., numpy.newaxis]
        vis_imaging_weight = (
            vis.visibility_acc.flagged_imaging_weight[..., 0]
            + vis.visibility_acc.flagged_imaging_weight[..., 1]
        )[..., numpy.newaxis]
    else:
        raise NameError(f"Polarisation frame {poldef} unknown")

    new_vis = Visibility.constructor(
        frequency=vis.frequency.data,
        channel_bandwidth=vis.channel_bandwidth.data,
        phasecentre=vis.phasecentre,
        baselines=vis["baselines"],
        configuration=vis.attrs["configuration"],
        uvw=vis["uvw"].data,
        time=vis["time"].data,
        vis=vis_data,
        flags=vis_flags,
        weight=vis_weight,
        integration_time=vis["integration_time"].data,
        polarisation_frame=polarisation_frame,
        source=vis.attrs["source"],
        meta=vis.attrs["meta"],
    )
    new_vis.imaging_weight = vis_imaging_weight
    return new_vis


def convert_visibility_stokesI_to_polframe(vis, poldef=None):
    """Convert the Stokes I into full polarisation, return new Visibility.

    :param vis: Visibility
    :param poldef: Desired polarisation frame
    :return: Converted visibility data.
    """
    if vis.visibility_acc.polarisation_frame == poldef:
        return vis

    npol = poldef.npol

    stokesvis = vis.visibility_acc.flagged_vis[..., 0][..., numpy.newaxis]
    vis_data = numpy.repeat(stokesvis, npol, axis=-1)

    stokesflags = vis.flags.data[..., 0][..., numpy.newaxis]
    vis_flags = numpy.repeat(stokesflags, npol, axis=-1)

    stokesweight = vis.visibility_acc.flagged_weight[..., 0][
        ..., numpy.newaxis
    ]
    vis_weight = numpy.repeat(stokesweight, npol, axis=-1)

    stokesimaging_weight = vis.visibility_acc.flagged_imaging_weight[..., 0][
        ..., numpy.newaxis
    ]
    vis_imaging_weight = numpy.repeat(stokesimaging_weight, npol, axis=-1)

    vis_data[..., 1] = 0.0
    vis_data[..., 2] = 0.0

    new_vis = Visibility.constructor(
        frequency=vis.frequency.data,
        channel_bandwidth=vis.channel_bandwidth.data,
        phasecentre=vis.phasecentre,
        baselines=vis["baselines"],
        configuration=vis.attrs["configuration"],
        uvw=vis["uvw"].data,
        time=vis["time"].data,
        vis=vis_data,
        flags=vis_flags,
        weight=vis_weight,
        integration_time=vis["integration_time"].data,
        polarisation_frame=poldef,
        source=vis.attrs["source"],
        meta=vis.attrs["meta"],
    )
    new_vis.imaging_weight = vis_imaging_weight
    return new_vis


def expand_polarizations(data, dtype=None):
    """
    Expand number of polarizations to four
    Optionally change the data type

    :param data: numpy array containing visibility data. It has dimensions
                (frequency, baselines, polarizations)
    :param dtype: optional data type
    :return: numpy array containing the input visibility data, where the
            polarizations dimension is fixed to 4 (data is copied)
    """

    if dtype is None:
        dtype = data.dtype
    nr_polarizations = data.shape[-1]
    if (nr_polarizations == 4) and (dtype == data.dtype):
        # Nothing to do
        # Return unmodified input
        return data
    new_shape = data.shape[:-1] + (4,)
    data_out = numpy.zeros(new_shape, dtype=dtype)
    if nr_polarizations == 4:
        data_out[:] = data
    elif nr_polarizations == 2:
        data_out[:, :, 0] = data[:, :, 0]
        data_out[:, :, 3] = data[:, :, 1]
    else:
        data_out[:, :, 0] = data[:, :, 0]
        data_out[:, :, 3] = data[:, :, 0]
    return data_out


def copy_data_and_shrink_polarizations(data, nr_polarizations, dtype=None):
    """
    Shrink number of polarizations to nr_polarizations
    Optionally change the data type

    :param data: numpy array containing visibility data. It should have
     dimensions (frequency, baselines, 4)
    :param nr_polarizations: number of polarizations desired in the output.
        Can be 1, 2 or 4
    :param dtype: optional data type
    :return: numpy array containing the input visibility data, where the
            polarizations dimension is nr_polarizations (data is copied)
    """

    if dtype is None:
        dtype = data.dtype

    new_shape = data.shape[:-1] + (nr_polarizations,)
    data_out = numpy.zeros(new_shape, dtype=dtype)

    if nr_polarizations == 4:
        data_out[:] = data

    elif nr_polarizations == 2:
        # If there are two polarizations, they are XX and YY
        data_out[:, :, 0] = data[:, :, 0]
        data_out[:, :, 1] = data[:, :, 3]

    else:
        # If there is a single polarization, it is the average of XX and YY
        data_out[:, :, 0] = (data[:, :, 0] + data[:, :, 3]) / 2
    return data_out
