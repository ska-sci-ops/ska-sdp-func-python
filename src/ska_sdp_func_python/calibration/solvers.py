"""
Functions to solve for antenna/station gain.
"""

__all__ = ["solve_gaintable"]

import logging

import numpy
import scipy
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ska_sdp_func_python.visibility.operations import divide_visibility

log = logging.getLogger("func-python-logger")


def find_best_refant_from_vis(vis):
    """
    This method comes from katsdpcal.
    (https://github.com/ska-sa/katsdpcal/blob/
    200c2f6e60b2540f0a89e7b655b26a2b04a8f360/katsdpcal/calprocs.py#L332)
    Determine antenna whose FFT has the maximum peak to noise ratio (PNR) by
    taking the median PNR of the FFT over all baselines to each antenna.

    When the input vis has only one channel, this uses all the vis of the
    same antenna for the operations peak, mean and std.

    :param vis: Visibility containing the observed data_models
    :return: sorted refant array

    """
    visdata = vis.visibility_acc.flagged_vis
    _, _, nchan, _ = visdata.shape
    baselines = numpy.array(vis.baselines.data.tolist())
    nants = vis.visibility_acc.nants
    med_pnr_ants = numpy.zeros((nants))
    if nchan == 1:
        weightdata = vis.visibility_acc.flagged_weight
        for a in range(nants):
            mask = (baselines[:, 0] == a) ^ (baselines[:, 1] == a)
            weightdata_ant = weightdata[:, mask]
            mean_of_weight_ant = numpy.sum(weightdata_ant)
            med_pnr_ants[a] = mean_of_weight_ant
        med_pnr_ants += numpy.linspace(1e-8, 1e-9, nants)
    else:
        ft_vis = scipy.fftpack.fft(visdata, axis=2)
        k_arg = numpy.argmax(numpy.abs(ft_vis), axis=2)
        index = numpy.array(
            [numpy.roll(range(nchan), -n) for n in k_arg.ravel()]
        )
        index = index.reshape(list(k_arg.shape) + [nchan])
        index = numpy.transpose(index, (0, 1, 3, 2))
        ft_vis = numpy.take_along_axis(ft_vis, index, axis=2)

        peak = numpy.max(numpy.abs(ft_vis), axis=2)

        chan_slice = numpy.s_[
            nchan // 2 - nchan // 4 : nchan // 2 + nchan // 4 + 1
        ]
        mean = numpy.mean(numpy.abs(ft_vis[:, :, chan_slice]), axis=2)
        std = numpy.std(numpy.abs(ft_vis[:, :, chan_slice]), axis=2) + 1e-9
        for a in range(nants):
            mask = (baselines[:, 0] == a) ^ (baselines[:, 1] == a)

            pnr = (peak[:, mask] - mean[:, mask]) / std[:, mask]
            med_pnr = numpy.median(pnr)
            med_pnr_ants[a] = med_pnr
    return numpy.argsort(med_pnr_ants)[::-1]


def solve_gaintable(
    vis: Visibility,
    modelvis: Visibility = None,
    gain_table=None,
    phase_only=True,
    niter=200,
    tol=1e-6,
    crosspol=False,
    normalise_gains="mean",
    jones_type="T",
    timeslice=None,
    refant=0,
) -> GainTable:
    """
    Solve a gain table by fitting an observed visibility
    to a model visibility.

    If modelvis is None, a point source model is assumed.

    :param vis: Visibility containing the observed data_models
    :param modelvis: Visibility containing the visibility predicted by a model
    :param gain_table: Existing gaintable
    :param phase_only: Solve only for the phases (default=True)
    :param niter: Number of iterations (default 30)
    :param tol: Iteration stops when the fractional change
                 in the gain solution is below this tolerance
    :param crosspol: Do solutions including cross polarisations
                     i.e. XY, YX or RL, LR
    :param normalise_gains: Normalises the gains (default="mean")
                     options are None, "mean", "median".
                     None means no normalization.
    :param jones_type: Type of calibration matrix T or G or B
    :param timeslice: Time interval between solutions (s)
    :param refant: Phase alignment to its reference antenna (default 0)
    :return: GainTable containing solution

    """
    if modelvis is not None:
        # pylint: disable=unneeded-not
        if not numpy.max(numpy.abs(modelvis.vis)) > 0.0:
            raise ValueError("solve_gaintable: Model visibility is zero")

    point_vis = (
        divide_visibility(vis, modelvis) if modelvis is not None else vis
    )

    if phase_only:
        log.debug("solve_gaintable: Solving for phase only")
    else:
        log.debug("solve_gaintable: Solving for complex gain")

    if gain_table is None:
        log.debug("solve_gaintable: creating new gaintable")
        gain_table = create_gaintable_from_visibility(
            vis, jones_type=jones_type, timeslice=timeslice
        )
    else:
        log.debug("solve_gaintable: starting from existing gaintable")

    nants = gain_table.gaintable_acc.nants
    nchan = gain_table.gaintable_acc.nchan
    npol = point_vis.visibility_acc.npol

    axes = (0, 2) if nchan == 1 else 0

    for row, time in enumerate(gain_table.time):
        time_slice = {
            "time": slice(
                time - gain_table.interval[row] / 2,
                time + gain_table.interval[row] / 2,
            )
        }
        pointvis_sel = point_vis.sel(time_slice)
        # pylint: disable=unneeded-not
        if not pointvis_sel.visibility_acc.ntimes > 0:
            log.warning(
                "Gaintable %s, vis time mismatch %s", gain_table.time, vis.time
            )
            continue
        refant_sort = find_best_refant_from_vis(pointvis_sel)
        x_b = numpy.sum(
            (pointvis_sel.vis.data * pointvis_sel.weight.data)
            * (1 - pointvis_sel.flags.data),
            axis=axes,
        )
        xwt_b = numpy.sum(
            pointvis_sel.weight.data * (1 - pointvis_sel.flags.data),
            axis=axes,
        )
        x = numpy.zeros([nants, nants, nchan, npol], dtype="complex")
        xwt = numpy.zeros([nants, nants, nchan, npol])
        for ibaseline, (a1, a2) in enumerate(point_vis.baselines.data):
            x[a1, a2, ...] = numpy.conjugate(x_b[ibaseline, ...])
            xwt[a1, a2, ...] = xwt_b[ibaseline, ...]
            x[a2, a1, ...] = x_b[ibaseline, ...]
            xwt[a2, a1, ...] = xwt_b[ibaseline, ...]

        mask = numpy.abs(xwt) > 0.0
        if numpy.sum(mask) > 0:
            _solve_with_mask(
                crosspol,
                gain_table,
                mask,
                niter,
                phase_only,
                row,
                tol,
                vis,
                x,
                xwt,
                refant,
                refant_sort,
            )
        else:
            gain_table["gain"].data[row, ...] = 1.0 + 0.0j
            gain_table["weight"].data[row, ...] = 0.0
            gain_table["residual"].data[row, ...] = 0.0

    if normalise_gains in ["median", "mean"] and not phase_only:
        normaliser = {
            "median": numpy.median,
            "mean": numpy.mean,
        }
        gabs = normaliser[normalise_gains](
            numpy.abs(gain_table["gain"].data[:])
        )
        gain_table["gain"].data[:] /= gabs

    return gain_table


def _solve_with_mask(
    crosspol,
    gain_table,
    mask,
    niter,
    phase_only,
    row,
    tol,
    vis,
    x,
    xwt,
    refant,
    refant_sort,
):
    """
    Method extracted from solve_gaintable to decrease
    complexity. Calculations when `numpy.sum(mask) > 0`
    """
    x_shape = x.shape
    x[mask] = x[mask] / xwt[mask]
    x[~mask] = 0.0
    xwt[mask] = xwt[mask] / numpy.max(xwt[mask])
    xwt[~mask] = 0.0
    x = x.reshape(x_shape)
    if vis.visibility_acc.npol == 2 or (
        vis.visibility_acc.npol == 4 and not crosspol
    ):
        (
            gain_table["gain"].data[row, ...],
            gain_table["weight"].data[row, ...],
            gain_table["residual"].data[row, ...],
        ) = _solve_antenna_gains_itsubs_nocrossdata(
            gain_table["gain"].data[row, ...],
            gain_table["weight"].data[row, ...],
            x,
            xwt,
            phase_only=phase_only,
            niter=niter,
            tol=tol,
            refant=refant,
            refant_sort=refant_sort,
        )
    elif vis.visibility_acc.npol == 4 and crosspol:
        (
            gain_table["gain"].data[row, ...],
            gain_table["weight"].data[row, ...],
            gain_table["residual"].data[row, ...],
        ) = _solve_antenna_gains_itsubs_matrix(
            gain_table["gain"].data[row, ...],
            gain_table["weight"].data[row, ...],
            x,
            xwt,
            phase_only=phase_only,
            niter=niter,
            tol=tol,
            refant=refant,
            refant_sort=refant_sort,
        )

    else:
        (
            gain_table["gain"].data[row, ...],
            gain_table["weight"].data[row, ...],
            gain_table["residual"].data[row, ...],
        ) = _solve_antenna_gains_itsubs_scalar(
            gain_table["gain"].data[row, ...],
            gain_table["weight"].data[row, ...],
            x,
            xwt,
            phase_only=phase_only,
            niter=niter,
            tol=tol,
            refant=refant,
            refant_sort=refant_sort,
        )


def _solve_antenna_gains_itsubs_scalar(
    gain,
    gwt,
    x,
    xwt,
    niter=200,
    tol=1e-6,
    phase_only=True,
    damping=0.5,
    refant=0,
    refant_sort=None,
):
    """Solve for the antenna gains.

     x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

     This uses an iterative substitution algorithm due to Larry
     D'Addario c 1980'ish (see ThompsonDaddario1982 Appendix 1). Used
     in the original VLA Dec-10 Antsol.

    :param gain: gains
    :param gwt: gain weight
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0)
    :param refant_sort: Sorted list of reference antenna
    :param damping: Damping parameter
    :return: gain [nants, ...], weight [nants, ...]

    """
    if refant_sort is None:
        refant_sort = []
    nants = x.shape[0]
    # Optimized
    i_diag = numpy.diag_indices(nants, nants)
    x[i_diag[0], i_diag[1], ...] = 0.0
    xwt[i_diag[0], i_diag[1], ...] = 0.0
    i_lower = numpy.tril_indices(nants, -1)
    i_upper = (i_lower[1], i_lower[0])
    x[i_upper] = numpy.conjugate(x[i_lower])
    xwt[i_upper] = xwt[i_lower]

    reduce_oneside_x = numpy.abs(numpy.einsum("ij...->j...", x * xwt))
    gainmask = reduce_oneside_x <= 0.0

    bad_ant = []
    for iant in range(nants):
        thismask = gainmask[iant, 0]
        if numpy.all(thismask) is True:
            bad_ant.append(iant)

    if refant in bad_ant:
        thisrefant = refant

        for ant_id in refant_sort:
            if ant_id not in bad_ant:
                refant = ant_id
                log.warning(
                    "warning, ant: %d is masked, \
                    change refant to ant: %d",
                    thisrefant,
                    refant,
                )
                break
        else:
            log.warning(
                "warning, Cannot find a suitable reference antenna,\
                 use initial settings: %d",
                thisrefant,
            )
            refant = thisrefant

    numpy.putmask(gain, gainmask, 0.0)
    for _ in range(niter):
        gainLast = gain
        gain, gwt = _gain_substitution_scalar(gain, x, xwt)
        if phase_only:
            mask = numpy.abs(gain) > 0.0
            gain[mask] = gain[mask] / numpy.abs(gain[mask])
        gain = (1.0 - damping) * gain + damping * gainLast
        change = numpy.max(numpy.abs(gain - gainLast))
        if change < tol:
            if phase_only:
                mask = numpy.abs(gain) > 0.0
                gain[mask] = gain[mask] / numpy.abs(gain[mask])
            angles = numpy.angle(gain)
            gain *= numpy.exp(-1j * angles)[refant, ...]
            numpy.putmask(gain, gainmask, 1.0)
            return gain, gwt, _solution_residual_scalar(gain, x, xwt)

    log.warning(
        "solve_antenna_gains_itsubs_scalar: "
        "gain solution failed, retaining gain solutions"
    )

    if phase_only:
        mask = numpy.abs(gain) > 0.0
        gain[mask] = gain[mask] / numpy.abs(gain[mask])
    angles = numpy.angle(gain)
    gain *= numpy.exp(-1j * angles)[refant, ...]
    numpy.putmask(gain, gainmask, 1.0)
    return gain, gwt, _solution_residual_scalar(gain, x, xwt)


def _gain_substitution_scalar(gain, x, xwt):
    """
    Substitute gains across all baselines of gain
         for point source equivalent visibilities.
    TODO: Check this function description

    :param gain: gains (numpy.array of shape [nant, nchan, nrec, nrec])
    :param x: Equivalent point source visibility [nants, nants, nchan, npol]
    :param xwt: Equivalent point source weight [nants, nants, nchan]?
    :return: gain [nants, nchan, nrec, nrec], weight [nants, nchan, nrec, nrec]

    """
    nants, nchan, nrec, _ = gain.shape

    newgain1 = numpy.ones_like(gain, dtype="complex128")
    gwt1 = numpy.zeros_like(gain, dtype="double")

    xxwt = x * xwt[:, :, :]
    cgain = numpy.conjugate(gain)
    gcg = gain[:, :] * cgain[:, :]

    n_top = numpy.einsum("ik...,ijk...->jk...", gain, xxwt)
    n_bot = numpy.einsum("ik...,ijk...->jk...", gcg, xwt).real

    # Convert mask to putmask
    numpy.putmask(newgain1, n_bot > 0.0, n_top / n_bot)
    numpy.putmask(newgain1, n_bot <= 0.0, 0.0)

    gwt1[:, :] = n_bot
    numpy.putmask(gwt1, n_bot <= 0.0, 0.0)

    newgain1 = newgain1.reshape([nants, nchan, nrec, nrec])
    gwt1 = gwt1.reshape([nants, nchan, nrec, nrec])
    return newgain1, gwt1


def _solve_antenna_gains_itsubs_nocrossdata(
    gain,
    gwt,
    x,
    xwt,
    niter=200,
    tol=1e-6,
    phase_only=True,
    refant=0,
    refant_sort=None,
):
    """Solve for the antenna gains using full matrix expressions,
         but no cross hands.

     x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

     See Appendix D, section D.1 in:

     J. P. Hamaker, “Understanding radio polarimetry - IV.
     The full-coherency analogue of scalar self-calibration:
     Self-alignment, dynamic range and polarimetric fidelity,”
     Astronomy and Astrophysics Supplement Series, vol. 143,
     no. 3, pp. 515–534, May 2000.

    :param gain: gains
    :param gwt: gain weight
    :param x: Equivalent point source visibility [nants, nants, nchan, npol]
    :param xwt: Equivalent point source weight [nants, nants, nchan]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0)
    :param refant_sort: Sorted list of reference antenna
    :return: gain [nants, nchan, nrec, nrec], weight [nants, nchan, nrec, nrec]
    """

    # This implementation is sub-optimal. TODO: Reimplement IQ, IV calibration

    if refant_sort is None:
        refant_sort = []
    nants, _, nchan, npol = x.shape
    if npol == 2:
        newshape = (nants, nants, nchan, 4)
        x_fill = numpy.zeros(newshape, dtype="complex")
        x_fill[..., 0] = x[..., 0]
        x_fill[..., 3] = x[..., 1]
        xwt_fill = numpy.zeros(newshape, dtype="float")
        xwt_fill[..., 0] = xwt[..., 0]
        xwt_fill[..., 3] = xwt[..., 1]
    else:
        x_fill = x
        x_fill[..., 1] = 0.0
        x_fill[..., 2] = 0.0
        xwt_fill = xwt
        xwt_fill[..., 1] = 0.0
        xwt_fill[..., 2] = 0.0

    return _solve_antenna_gains_itsubs_matrix(
        gain,
        gwt,
        x_fill,
        xwt_fill,
        niter=niter,
        tol=tol,
        phase_only=phase_only,
        refant=refant,
        refant_sort=refant_sort,
    )


def _solve_antenna_gains_itsubs_matrix(
    gain,
    gwt,
    x,
    xwt,
    niter=200,
    tol=1e-6,
    phase_only=True,
    refant=0,
    refant_sort=None,
):
    """Solve for the antenna gains using full matrix expressions.

     x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

     See Appendix D, section D.1 in:

     J. P. Hamaker, “Understanding radio polarimetry -
     IV. The full-coherency analogue of scalar self-calibration:
     Self-alignment, dynamic range and polarimetric fidelity,”
     Astronomy and Astrophysics Supplement Series, vol. 143,
     no. 3, pp. 515–534, May 2000.

    :param gain: gains
    :param gwt: gain weight
    :param x: Equivalent point source visibility[nants, nants, nchan, npol]
    :param xwt: Equivalent point source weight [nants, nants, nchan]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0)
    :param refant_sort: Sorted list of reference antenna
    :return: gain [nants, nchan, nrec, nrec], weight [nants, nchan, nrec, nrec]
    """

    if refant_sort is None:
        refant_sort = []
    nants, _, nchan, npol = x.shape
    assert npol == 4
    newshape = (nants, nants, nchan, 2, 2)
    x = x.reshape(newshape)
    xwt = xwt.reshape(newshape)

    # Optimzied
    i_diag = numpy.diag_indices(nants, nants)
    x[i_diag[0], i_diag[1], ...] = 0.0
    xwt[i_diag[0], i_diag[1], ...] = 0.0
    i_lower = numpy.tril_indices(nants, -1)
    i_upper = (i_lower[1], i_lower[0])
    x[i_upper] = numpy.conjugate(x[i_lower])
    xwt[i_upper] = xwt[i_lower]

    gain[..., 0, 1] = 0.0
    gain[..., 1, 0] = 0.0

    reduce_oneside_x = numpy.abs(numpy.einsum("ij...->j...", x * xwt))
    gainmask = reduce_oneside_x <= 0.0

    # If the cross pol item is masked, its fallback value is 0
    cross_mask = gainmask.copy()
    cross_mask[..., 0, 0] = False
    cross_mask[..., 1, 1] = False
    cross_mask[..., 0, 1] = gainmask[..., 0, 1]
    cross_mask[..., 1, 0] = gainmask[..., 1, 0]

    bad_ant = []
    for iant in range(nants):
        # TODO The current judgment uses channel 0.
        # If all polarizations of this channel are marked,
        # the antenna is considered bad
        thismask = gainmask[iant, 0]
        if numpy.all(thismask) is True:
            bad_ant.append(iant)

    if refant in bad_ant:
        thisrefant = refant

        for ant_id in refant_sort:
            if ant_id not in bad_ant:
                refant = ant_id
                log.warning(
                    "warning, ant: %d is masked, \
                    change refant to ant: %d",
                    thisrefant,
                    refant,
                )
                break
        else:
            log.warning(
                "warning, Cannot find a suitable reference antenna,\
                 use initial settings: %d",
                thisrefant,
            )
            refant = thisrefant

    numpy.putmask(gain, gainmask, 0.0)
    for _ in range(niter):
        gainLast = gain
        gain, gwt = _gain_substitution_matrix(gain, x, xwt)
        if phase_only:
            mask = numpy.abs(gain) > 0.0
            gain[mask] = gain[mask] / numpy.abs(gain[mask])
        change = numpy.max(numpy.abs(gain - gainLast))
        gain = 0.5 * (gain + gainLast)
        if change < tol:
            angles = numpy.angle(gain)
            gain *= numpy.exp(-1j * angles)[refant, ...]
            numpy.putmask(gain, gainmask, 1.0)
            numpy.putmask(gain, cross_mask, 0.0)
            return gain, gwt, _solution_residual_matrix(gain, x, xwt)

    log.warning(
        "solve_antenna_gains_itsubs_scalar: "
        "gain solution failed, retaining gain solutions"
    )
    angles = numpy.angle(gain)
    gain *= numpy.exp(-1j * angles)[refant, ...]
    numpy.putmask(gain, gainmask, 1.0)
    numpy.putmask(gain, cross_mask, 0.0)
    return gain, gwt, _solution_residual_matrix(gain, x, xwt)


def _gain_substitution_matrix(gain, x, xwt):
    """
    Substitute gains across all baselines of gain
         for point source equivalent visibilities.
    TODO: Check this function description

    :param gain: gains (numpy.array of shape [nant, nchan, nrec, nrec])
    :param x: Equivalent point source visibility [nants, nants, nchan, npol]
    :param xwt: Equivalent point source weight [nants, nants, nchan]
    :return: gain [nants, nchan, nrec, nrec], weight [nants, nchan, nrec, nrec]
    """
    nants, nchan, nrec, _ = gain.shape

    # We are going to work with Jones 2x2 matrix formalism
    # so everything has to be converted to that format
    x = x.reshape([nants, nants, nchan, nrec, nrec])
    diag = numpy.ones_like(x)
    xwt = xwt.reshape([nants, nants, nchan, nrec, nrec])
    # Write these loops out explicitly.
    # Derivation of these vector equations is tedious but they are
    # structurally identical to the scalar case with the following changes
    # Vis -> 2x2 coherency vector, g-> 2x2 Jones matrix,
    # *-> matmul, conjugate->Hermitean transpose (.H)
    gain_conj = numpy.conjugate(gain)
    for ant in range(nants):
        diag[ant, ant, ...] = 0
    n_top1 = numpy.einsum("ij...->j...", xwt * diag * x * gain[:, None, ...])
    n_bot = diag * xwt * gain_conj * gain
    n_bot1 = numpy.einsum("ij...->i...", n_bot)

    # Using putmask: faster than using Boolen Index
    n_top2 = n_top1.copy()
    numpy.putmask(n_top2, n_bot1[...] <= 0, 0.0)
    n_bot2 = n_bot1.copy()
    numpy.putmask(n_bot2, n_bot1[...] <= 0, 1.0)
    newgain1 = n_top2 / n_bot2

    gwt1 = n_bot1.real
    return newgain1, gwt1


def _solution_residual_scalar(gain, x, xwt):
    """Calculate residual across all baselines of gain
         for point source equivalent visibilities.

    :param gain: gains (numpy.array of shape [nant, nchan, nrec, nrec])
    :param x: Equivalent point source visibility [nants, nants, nchan, npol]
    :param xwt: Equivalent point source weight [nants, nants, nchan]
    :return: residual[nchan, nrec, nrec]
    """

    nant, nchan, nrec, _ = gain.shape
    x = x.reshape(nant, nant, nchan, nrec, nrec)

    xwt = xwt.reshape(nant, nant, nchan, nrec, nrec)

    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])

    for chan in range(nchan):
        lgain = gain[:, chan, 0, 0]
        clgain = numpy.conjugate(lgain)
        smueller = numpy.ma.outer(clgain, lgain).reshape([nant, nant])
        error = x[:, :, chan, 0, 0] - smueller
        for i in range(nant):
            error[i, i] = 0.0
        residual[chan] += numpy.sum(
            error * xwt[:, :, chan, 0, 0] * numpy.conjugate(error)
        ).real
        sumwt[chan] += numpy.sum(xwt[:, :, chan, 0, 0])

    residual[sumwt > 0.0] = numpy.sqrt(
        residual[sumwt > 0.0] / sumwt[sumwt > 0.0]
    )
    residual[sumwt <= 0.0] = 0.0

    return residual


def _solution_residual_matrix(gain, x, xwt):
    """Calculate residual across all baselines of gain
         for point source equivalent visibilities.

    :param gain: gains (numpy.array of shape [nant, nchan, nrec, nrec])
    :param x: Equivalent point source visibility [nants, nants, nchan, npol]
    :param xwt: Equivalent point source weight [nants, nants, nchan]
    :return: residual[nchan, nrec, nrec]
    """
    n_gain = numpy.einsum("i...,j...->ij...", numpy.conjugate(gain), gain)
    n_error = numpy.conjugate(x - n_gain)
    nn_residual = (n_error * xwt * numpy.conjugate(n_error)).real
    n_residual = numpy.einsum("ijk...->k...", nn_residual)
    n_sumwt = numpy.einsum("ijk...->k...", xwt)

    n_residual[n_sumwt > 0.0] = numpy.sqrt(
        n_residual[n_sumwt > 0.0] / n_sumwt[n_sumwt > 0.0]
    )
    n_residual[n_sumwt <= 0.0] = 0.0

    return n_residual
