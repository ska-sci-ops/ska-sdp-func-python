"""
Functions for calibration.
"""

__all__ = [
    "apply_gaintable",
    "multiply_gaintables",
    "concatenate_gaintables",
]

import copy
import logging

import numpy.linalg
import xarray
from astropy.time import Time
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.visibility.vis_model import Visibility

log = logging.getLogger("func-python-logger")


def apply_gaintable(
    vis: Visibility, gt: GainTable, inverse=False, **kwargs
) -> Visibility:
    """Apply a gain table to a visibility

    The corrected visibility is::

        V_corrected = {g_i * g_j^*}^-1 V_obs

    If the visibility data are polarised
    e.g. polarisation_frame("linear") then the inverse operator
    represents an actual inverse of the gains.

    :param vis: visibility to have gains applied
    :param gt: Gaintable to be applied
    :param inverse: Apply the inverse (default=False)
    :param use_flags: Use flags? (in kwargs)
    :return: input vis with gains applied

    """
    ntimes, nants, nchan, nrec, _ = gt.gain.shape

    if inverse:
        log.debug("apply_gaintable: Apply inverse gaintable")
    else:
        log.debug("apply_gaintable: Apply gaintable")

    if vis.visibility_acc.npol == 1:
        log.debug("apply_gaintable: scalar gains")

    row_numbers = numpy.arange(len(vis.time))

    for row in range(ntimes):
        vis_rows = (
            numpy.abs(vis.time.data - gt.time.data[row])
            < gt.interval.data[row] / 2.0
        )
        vis_rows = row_numbers[vis_rows]
        if len(vis_rows) > 0:

            # Lookup the gain for this set of visibilities
            gain = gt["gain"].data[row]
            cgain = numpy.conjugate(gt["gain"].data[row])

            # The shape of the mueller matrix is
            nant, nchan, nrec, _ = gain.shape
            baselines = vis.baselines.data

            # Try to ignore visibility flags in application of gains.
            # Should have no impact and will save time in applying the flags
            use_flags = kwargs["use_flags"]
            flagged = (
                use_flags and numpy.max(vis["flags"][vis_rows].data) > 0.0
            )
            if flagged:
                log.debug("apply_gaintable:Applying flags")
                original = vis.visibility_acc.flagged_vis[vis_rows]
                applied = copy.deepcopy(original)
                appliedwt = copy.deepcopy(
                    vis.visibility_acc.flagged_weight[vis_rows]
                )
            else:
                log.debug("apply_gaintable:flags are absent or being ignored")
                original = vis["vis"].data[vis_rows]
                applied = copy.deepcopy(original)
                appliedwt = copy.deepcopy(vis["weight"].data[vis_rows])

            if vis.visibility_acc.npol == 1:
                if inverse:
                    lgain = numpy.zeros_like(gain)
                    try:
                        numpy.putmask(lgain, numpy.abs(gain) > 0.0, 1.0 / gain)
                    except FloatingPointError:
                        pass
                else:
                    lgain = gain

                # Optimized (SIM-423)
                # smueller1 = numpy.ones([nchan, nant, nant], dtype='complex')
                smueller1 = numpy.einsum(
                    "ijlm,kjlm->jik", lgain, numpy.conjugate(lgain)
                )

                for sub_vis_row in range(original.shape[0]):
                    for ibaseline, (a1, a2) in enumerate(baselines):
                        for chan in range(nchan):
                            if numpy.abs(smueller1[chan, a1, a2]) > 0.0:
                                applied[sub_vis_row, ibaseline, chan, 0] = (
                                    original[sub_vis_row, ibaseline, chan, 0]
                                    * smueller1[chan, a1, a2]
                                )
                            else:
                                applied[sub_vis_row, ibaseline, chan, 0] = 0.0
                                appliedwt[
                                    sub_vis_row, ibaseline, chan, 0
                                ] = 0.0

            elif vis.visibility_acc.npol == 2:
                has_inverse_ant = numpy.zeros([nant, nchan], dtype="bool")
                if inverse:
                    igain = gain.copy()
                    cigain = cgain.copy()
                    for a1 in range(nants):
                        for chan in range(nchan):
                            try:
                                igain[a1, chan, :, :] = numpy.linalg.inv(
                                    gain[a1, chan, :, :]
                                )
                                cigain[a1, chan, :, :] = numpy.conjugate(
                                    igain[a1, chan, :, :]
                                )
                                has_inverse_ant[a1, chan] = True
                            except numpy.linalg.LinAlgError:
                                has_inverse_ant[a1, chan] = False

                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, (a1, a2) in enumerate(baselines):
                            for chan in range(nchan):
                                if (
                                    has_inverse_ant[a1, chan]
                                    and has_inverse_ant[a2, chan]
                                ):
                                    cfs = numpy.diag(
                                        original[
                                            sub_vis_row, ibaseline, chan, ...
                                        ]
                                    )
                                    applied[
                                        sub_vis_row, ibaseline, chan, ...
                                    ] = numpy.diag(
                                        igain[a1, chan, :, :]
                                        @ cfs
                                        @ cigain[a2, chan, :, :]
                                    ).reshape(
                                        [2]
                                    )
                                else:
                                    applied[
                                        sub_vis_row, ibaseline, chan, 0
                                    ] = 0.0
                                    appliedwt[
                                        sub_vis_row, ibaseline, chan, 0
                                    ] = 0.0

                else:
                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, (a1, a2) in enumerate(baselines):
                            for chan in range(nchan):
                                cfs = numpy.diag(
                                    original[sub_vis_row, ibaseline, chan, ...]
                                )
                                applied[
                                    sub_vis_row, ibaseline, chan, ...
                                ] = numpy.diag(
                                    gain[a1, chan, :, :]
                                    @ cfs
                                    @ cgain[a2, chan, :, :]
                                ).reshape(
                                    [2]
                                )

            elif vis.visibility_acc.npol == 4:
                has_inverse_ant = numpy.zeros([nant, nchan], dtype="bool")
                if inverse:
                    igain = gain.copy()
                    cigain = cgain.copy()
                    for a1 in range(nants):
                        for chan in range(nchan):
                            try:
                                igain[a1, chan, :, :] = numpy.linalg.inv(
                                    gain[a1, chan, :, :]
                                )
                                cigain[a1, chan, :, :] = numpy.conjugate(
                                    igain[a1, chan, :, :]
                                )
                                has_inverse_ant[a1, chan] = True
                            except numpy.linalg.LinAlgError:
                                has_inverse_ant[a1, chan] = False

                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, baseline in enumerate(baselines):
                            for chan in range(nchan):
                                if (
                                    has_inverse_ant[baseline[0], chan]
                                    and has_inverse_ant[baseline[1], chan]
                                ):
                                    cfs = original[
                                        sub_vis_row, ibaseline, chan, ...
                                    ].reshape([2, 2])
                                    applied[
                                        sub_vis_row, ibaseline, chan, ...
                                    ] = (
                                        igain[baseline[0], chan, :, :]
                                        @ cfs
                                        @ cigain[baseline[1], chan, :, :]
                                    ).reshape(
                                        [4]
                                    )
                                else:
                                    applied[
                                        sub_vis_row, ibaseline, chan, ...
                                    ] = 0.0
                                    appliedwt[
                                        sub_vis_row, ibaseline, chan, ...
                                    ] = 0.0
                else:
                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, baseline in enumerate(baselines):
                            for chan in range(nchan):
                                cfs = original[
                                    sub_vis_row, ibaseline, chan, ...
                                ].reshape([2, 2])
                                applied[sub_vis_row, ibaseline, chan, ...] = (
                                    gain[baseline[0], chan, :, :]
                                    @ cfs
                                    @ cgain[baseline[1], chan, :, :]
                                ).reshape([4])

            else:
                times = Time(vis.time / 86400.0, format="mjd", scale="utc")
                log.warning(
                    "No row in gaintable for visibility "
                    "row, time range  {} to {}".format(
                        times[0].isot, times[-1].isot
                    )
                )

            vis["vis"].data[vis_rows] = applied
            vis["weight"].data[vis_rows] = appliedwt

    return vis


def multiply_gaintables(
    gt: GainTable, dgt: GainTable, time_tolerance=1e-3
) -> GainTable:
    """Multiply two GainTables

    Returns gt * dgt

    :param gt: First GainTable
    :param dgt: Second GainTable
    :param time_tolerance: Maximum tolerance of time s
                eparation in the GainTable data
    :return: Multiplication product
    """

    # Test if times align
    mismatch = numpy.max(numpy.abs(gt["time"].data - dgt["time"].data))
    if mismatch > time_tolerance:
        raise ValueError(
            f"Gaintables not aligned in time: max mismatch {mismatch} seconds"
        )
    if dgt.gaintable_acc.nrec == gt.gaintable_acc.nrec:
        if dgt.gaintable_acc.nrec == 2:
            gt["gain"].data = numpy.einsum(
                "...ik,...ij->...kj", gt["gain"].data, dgt["gain"].data
            )
            gt["weight"].data *= dgt["weight"].data
        elif dgt.gaintable_acc.nrec == 1:
            gt["gain"].data *= dgt["gain"].data
            gt["weight"].data *= dgt["weight"].data
        else:
            raise ValueError(
                "Gain tables have illegal structures {} {}".format(
                    str(gt), str(dgt)
                )
            )

    else:
        raise ValueError(
            "Gain tables have different structures {} {}".format(
                str(gt), str(dgt)
            )
        )

    return gt


def concatenate_gaintables(gt_list, dim="time"):
    """Concatenate a list of GainTables

    :param gt_list: List of GainTables
    :param dim: Dimension to concatenate
    :return: Concatenated GainTable
    """

    if len(gt_list) == 0:
        raise ValueError("GainTable list is empty")

    return xarray.concat(
        gt_list,
        dim=dim,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
