"""
Functions to solve for antenna/station gain.
"""
# I will fix these if I can before the MR
# pylint: disable=too-many-lines
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=invalid-name
# if I fix these, black complains...
# flake8: noqa E203

__all__ = ["solve_gaintable"]

import logging

import numpy
import scipy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsmr
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ska_sdp_func_python.calibration.solver_utils import (
    gen_cdm,
    gen_coherency_products,
    gen_pol_matrix,
    update_design_matrix,
)
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

    :param vis: Visibilities
    :return: Array of indices of antennas in decreasing order
            of median of PNR over all baselines

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
        max_value_arg = numpy.argmax(numpy.abs(ft_vis), axis=2)
        index = numpy.array(
            [numpy.roll(range(nchan), -n) for n in max_value_arg.ravel()]
        )
        index = index.reshape(list(max_value_arg.shape) + [nchan])
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
    solver="gain_substitution",
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
        i.e. XY, YX or RL, LR. Only used by the gain_substitution solver.
    :param normalise_gains: Normalises the gains (default="mean")
        options are None, "mean", "median".
        None means no normalization.
    :param solver: Calibration algorithm to use (default="gain_substitution")
        options are:
        "gain_substitution" - original substitution algorithm with separate
            solutions for each polarisation term.
        "jones_substitution" - solve antenna-based Jones matrices as a whole,
            with independent updates within each iteration.
        "normal_equations" - solve normal equations within each iteration
            formed from linearisation with respect to antenna-based gain and
            leakage terms.
        "normal_equations_presum" - the same as the normal_equations option
            but with an initial accumulation of visibility products over time
            and frequency for each solution interval. This can be much faster
            for large datasets and solution intervals.
    :param jones_type: Type of calibration matrix T or G or B
    :param timeslice: Time interval between solutions (s)
    :param refant: Reference antenna (default 0). Currently only activated for
        the gain_substitution solver.
    :return: GainTable containing solution

    """
    if modelvis is not None:
        # pylint: disable=unneeded-not
        if not numpy.max(numpy.abs(modelvis.vis)) > 0.0:
            raise ValueError("solve_gaintable: Model visibility is zero")

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
    npol = vis.visibility_acc.npol

    axes = (0, 2) if nchan == 1 else 0

    # disable this if not needed by the solver
    pointvis = (
        divide_visibility(vis, modelvis) if modelvis is not None else vis
    )

    # moved this here so it doesn't change with time slice...
    refant_sort = find_best_refant_from_vis(pointvis)

    for row, time in enumerate(gain_table.time):
        time_slice = {
            "time": slice(
                time - gain_table.interval[row] / 2,
                time + gain_table.interval[row] / 2,
            )
        }
        vis_sel = vis.sel(time_slice)
        # pylint: disable=unneeded-not
        if not vis_sel.visibility_acc.ntimes > 0:
            log.warning(
                "Gaintable %s, vis time mismatch %s", gain_table.time, vis.time
            )
            continue

        # the gain_substitution and normal_equations solvers require that the
        # observed and model visibilities are kept separate.

        if solver == "jones_substitution":
            jones_sub_solve(
                vis_sel,
                modelvis.sel(time_slice),
                gain_table,
                niter,
                row,
                tol,
                refant,
            )

        elif solver == "normal_equations":
            _normal_equation_solve(
                vis_sel,
                modelvis.sel(time_slice),
                gain_table,
                niter,
                row,
                tol,
                refant,
            )

        elif solver == "normal_equations_presum":
            _normal_equation_solve_with_presumming(
                vis_sel,
                modelvis.sel(time_slice),
                gain_table,
                niter,
                row,
                tol,
                refant,
            )

        elif solver == "gain_substitution":
            # form intermediate arrays for the _solve_with_mask solver

            pointvis_sel = pointvis.sel(time_slice)

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
            for ibaseline, (a1, a2) in enumerate(pointvis_sel.baselines.data):
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
        else:
            log.warning("solve_gaintable: unknown solver: %s", solver)
            break

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


def jones_sub_solve(
    vis: Visibility,
    modelvis: Visibility,
    gain_table,
    niter,
    row,
    tol,
    refant,  # pylint: disable=unused-argument
):
    """
    Solve this row (time slice) of the gain table
    """
    gain = gain_table["gain"].data[row, ...]
    nants, nchan_gt, nrec1, nrec2 = gain.shape
    ntime, nbl, nchan_vis, _ = vis.vis.shape
    assert nrec1 == 2
    assert nrec1 == nrec2
    assert nchan_gt in (1, nchan_vis)
    chgt = numpy.zeros(nchan_vis, "int") if nchan_gt == 1 else range(nchan_vis)

    ant1 = vis.antenna1.data
    ant2 = vis.antenna2.data

    # to do: should the nchan_gt loop be outside the niter loop?

    for it in range(niter):
        gainLast = gain.copy()
        update = _jones_substitution(
            gain, vis.vis.data, modelvis.vis.data, vis.weight.data, ant1, ant2
        )

        # nu = 0.5
        nu = 1.0 - 0.5 * (it % 2)

        for ant in range(nants):
            for ch in range(nchan_gt):
                update[ant, ch] = numpy.eye(2) + nu * (
                    update[ant, ch] - numpy.eye(2)
                )
                gain[ant, ch] = update[ant, ch] @ gain[ant, ch]

        vmdl = modelvis.vis.data.reshape(ntime, nbl, nchan_vis, 2, 2)

        for t in range(ntime):
            for k in range(nbl):
                for f in range(nchan_vis):
                    vmdl[t, k, f] = (
                        update[ant1[k], chgt[f]]
                        @ vmdl[t, k, f]
                        @ update[ant2[k], chgt[f]].conj().T
                    )

        change = numpy.max(numpy.abs(gain - gainLast))

        if change < tol:
            # angles = numpy.angle(gain)
            # gain *= numpy.exp(-1j * angles)[refant, ...]
            return

    log.warning("jones_sub_solve: gain solution failed to converge")


def _normal_equation_solve(
    vis: Visibility,
    modelvis: Visibility,
    gain_table,
    niter,
    row,
    tol,
    refant,  # pylint: disable=unused-argument
):
    """
    Solve this row (time slice) of the gain table
    """
    gain = gain_table["gain"].data[row, ...]
    nants, nchan_gt, nrec1, nrec2 = gain.shape
    ntime, nbl, nchan_vis, _ = vis.vis.shape
    assert nrec1 == 2
    assert nrec1 == nrec2
    assert nchan_gt in (1, nchan_vis)
    chgt = numpy.zeros(nchan_vis, "int") if nchan_gt == 1 else range(nchan_vis)

    # convert Jones matrices to gain and leakage terms
    gX = numpy.zeros((nants, nchan_gt), "complex")
    gY = numpy.zeros((nants, nchan_gt), "complex")
    dXY = numpy.zeros((nants, nchan_gt), "complex")
    dYX = numpy.zeros((nants, nchan_gt), "complex")
    for ant in range(nants):
        for ch in range(nchan_gt):
            gX[ant, ch] = gain[ant, ch, 0, 0]
            gY[ant, ch] = gain[ant, ch, 1, 1]
            dXY[ant, ch] = gain[ant, ch, 0, 1] / gain[ant, ch, 0, 0]
            dYX[ant, ch] = -gain[ant, ch, 1, 0] / gain[ant, ch, 1, 1]

    ant1 = vis.antenna1.data
    ant2 = vis.antenna2.data

    # to do: include weighting (e.g. as diagonal matrix W: A.T @ W @ A)
    # to do: move the nchan_gt loop outside the niter loop

    vmdl0 = modelvis.vis.data.reshape(ntime, nbl, nchan_vis, 2, 2)

    # vmdl = numpy.empty(vmdl0.shape, "complex")
    vmdl = vmdl0.copy()

    for _ in range(niter):
        # multiply model by the current gain estimates
        for t in range(ntime):
            for k in range(nbl):
                for f in range(nchan_vis):
                    vmdl[t, k, f] = (
                        gain[ant1[k], chgt[f]]
                        @ vmdl0[t, k, f]
                        @ gain[ant2[k], chgt[f]].conj().T
                    )

        update = _calc_and_solve_normal_equations(
            gX,
            gY,
            dXY,
            dYX,
            vis.vis.data,
            vmdl,
            vis.weight.data,
            ant1,
            ant2,
        )

        # nu = 0.5
        nu = 1.0

        gX += nu * update[0]
        gY += nu * update[1]
        dXY += nu * update[2]
        dYX += nu * update[3]

        # update jones
        gainLast = gain.copy()
        for ant in range(nants):
            for ch in range(nchan_gt):
                gain[ant, ch] = [
                    [gX[ant, ch], gX[ant, ch] * dXY[ant, ch]],
                    [-gY[ant, ch] * dYX[ant, ch], gY[ant, ch]],
                ]

        change = numpy.max(numpy.abs(gain - gainLast))

        if change < tol:
            # angles = numpy.angle(gain)
            # gain *= numpy.exp(-1j * angles)[refant, ...]
            return

    log.warning("_normal_equation_solve: gain solution failed to converge")


def _normal_equation_solve_with_presumming(
    vis: Visibility,
    modelvis: Visibility,
    gain_table,
    niter,
    row,
    tol,
    refant,  # pylint: disable=unused-argument
):
    """
    Solve this row (time slice) of the gain table
    """
    gain = gain_table["gain"].data[row, ...]
    nants, nchan_gt, nrec1, nrec2 = gain.shape
    ntime, nbl, nchan_vis, _ = vis.vis.shape
    assert nrec1 == 2
    assert nrec1 == nrec2
    assert nchan_gt in (1, nchan_vis)

    ant1 = vis.antenna1.data
    ant2 = vis.antenna2.data

    for ch in range(nchan_gt):
        # convert Jones matrices to gain and leakage terms
        gX = numpy.zeros(nants, "complex")
        gY = numpy.zeros(nants, "complex")
        dXY = numpy.zeros(nants, "complex")
        dYX = numpy.zeros(nants, "complex")
        for ant in range(nants):
            gX[ant] = gain[ant, ch, 0, 0]
            gY[ant] = gain[ant, ch, 1, 1]
            dXY[ant] = gain[ant, ch, 0, 1] / gain[ant, ch, 0, 0]
            dYX[ant] = -gain[ant, ch, 1, 0] / gain[ant, ch, 1, 1]

        # select channels to average over. Just the current one if solving
        # each channel separately, or all of them if this is a joint solution.
        chan_vis = [ch] if nchan_gt == nchan_vis else range(nchan_vis)

        print(f"processing gain channel {ch} and averaging over {chan_vis}")

        # initialise pre-sum accumulation matrices
        Smo = numpy.zeros((nbl, 4, 4), "complex")
        Smm = numpy.zeros((nbl, 4, 4), "complex")
        for t in range(ntime):
            for k in range(nbl):
                for f in chan_vis:
                    mdlVec = modelvis.vis.data[t, k, f, :][numpy.newaxis, :]
                    obsVec = vis.vis.data[t, k, f, :][numpy.newaxis, :]
                    wgt = vis.weight.data[t, k, f, 0]
                    Smo[k] += wgt * mdlVec.conj().T @ obsVec
                    Smm[k] += wgt * mdlVec.conj().T @ mdlVec

        for it in range(niter):
            update = _calc_and_solve_normal_equations_with_presumming(
                gX,
                gY,
                dXY,
                dYX,
                Smo,
                Smm,
                ant1,
                ant2,
            )

            # nu = 0.5
            nu = 1.0

            gX += nu * update[0]
            gY += nu * update[1]
            dXY += nu * update[2]
            dYX += nu * update[3]

            # update jones
            gainLast = gain.copy()
            for ant in range(nants):
                gain[ant, ch] = [
                    [gX[ant], gX[ant] * dXY[ant]],
                    [-gY[ant] * dYX[ant], gY[ant]],
                ]

            change = numpy.max(numpy.abs(gain - gainLast))

            if change < tol:
                # angles = numpy.angle(gain)
                # gain *= numpy.exp(-1j * angles)[refant, ...]
                break

            if it == niter - 1:
                log.warning(
                    "_normal_equation_solve gain channel %d: \
                    solution failed to converge",
                    ch,
                )
                print(
                    "_normal_equation_solve gain channel %d: \
                    solution failed to converge",
                    ch,
                )


def _determine_refant(refant, bad_ant, refant_sort):
    """
    Determine the final reference antenna
    :param refant: the given reference antenna
    :param bad_ant: a list including all bad antennas
    :param refant_sort: a list with the decrease order of
                        reference antenna
    :return reference antenna
    """

    if refant in bad_ant:
        # Keep the original value of refant
        thisrefant = refant

        for ant_id in refant_sort:
            if ant_id not in bad_ant:
                # fetch an antenna from the refant_sort list and further judge
                # if the antenna is not a bad antenna, if Yes, this antenna
                # would be the final reference antenna
                refant = ant_id
                log.warning(
                    "warning, ant: %d is masked, \
                    change refant to ant: %d",
                    thisrefant,
                    refant,
                )
                break
        else:
            # If we cannot find a reference antenna, we have to use
            # the original refant kept in thisrefant
            log.warning(
                "warning, Cannot find a suitable reference antenna,\
                 use initial settings: %d",
                thisrefant,
            )
            refant = thisrefant


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

    _determine_refant(refant, bad_ant, refant_sort)

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
        # The current judgment uses channel 0.
        # If all polarizations of this channel are masked,
        # the antenna is considered bad
        thismask = gainmask[iant, 0]
        if numpy.all(thismask) is True:
            bad_ant.append(iant)

    _determine_refant(refant, bad_ant, refant_sort)

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


def _jones_substitution(gain, vis, modelvis, wgt, ant1, ant2):
    """
    Substitute gains across all baselines of gain
         for point source equivalent visibilities.
    TODO: Check this function description

    :param gain: gains (numpy.array of shape [nant, nchan, nrec, nrec])
    :param vis: Visibility containing the observed data_models
    :param modelvis: Visibility containing the visibility predicted by a model
    :return: gain [nants, nchan, nrec, nrec], weight [nants, nchan, nrec, nrec]
    """
    nants, nchan_gt, nrec1, nrec2 = gain.shape
    ntime, nbl, nchan_vis, _ = vis.shape
    # number of output gain channels can be 1 in total or 1 per vis channel
    chgt = numpy.zeros(nchan_vis, "int") if nchan_gt == 1 else range(nchan_vis)

    vobs = vis.reshape(ntime, nbl, nchan_vis, nrec1, nrec2)
    vmdl = modelvis.reshape(ntime, nbl, nchan_vis, nrec1, nrec2)

    Som = numpy.zeros((nants, nchan_gt, nrec1, nrec2), "complex")
    Smm = numpy.zeros((nants, nchan_gt, nrec1, nrec2), "complex")

    # assuming that the polarisations have the same weight
    # assert numpy.allclose(wgt[..., 0], wgt[..., 3])

    # The following is easier to read but not very efficient

    # for t in range(ntime):
    #     for k in range(nbl):
    #         if ant1[k] == ant2[k]:
    #             continue
    #         for f in range(nchan_vis):
    #             ch = chgt[f]
    #             # update sums for station 1
    #             Som[ant1[k], ch] += (
    #                 wgt[t, k, f, 0]
    #                 * vobs[t, k, f, :, :]
    #                 @ vmdl[t, k, f, :, :].conj().T
    #             )
    #             Smm[ant1[k], ch] += (
    #                 wgt[t, k, f, 0]
    #                 * vmdl[t, k, f, :, :]
    #                 @ vmdl[t, k, f, :, :].conj().T
    #             )
    #             # update sums for station 2
    #             Som[ant2[k], ch] += (
    #                 wgt[t, k, f, 0]
    #                 * vobs[t, k, f, :, :].conj().T
    #                 @ vmdl[t, k, f, :, :]
    #             )
    #             Smm[ant2[k], ch] += (
    #                 wgt[t, k, f, 0]
    #                 * vmdl[t, k, f, :, :].conj().T
    #                 @ vmdl[t, k, f, :, :]
    #             )

    # The following may be more efficient when there are many loops above

    for ant in range(nants):
        if nchan_gt == 1:
            # send all vis frequency channels to be accumulated
            gen_coherency_products(
                Som[ant, 0], Smm[ant, 0], vobs, vmdl, wgt, ant, ant1, ant2
            )
        else:
            # accumulate each vis frequency channel separately
            for f in range(nchan_vis):
                ch = chgt[f]
                gen_coherency_products(
                    Som[ant, ch],
                    Smm[ant, ch],
                    vobs[:, :, [f]],
                    vmdl[:, :, [f]],
                    wgt[:, :, [f]],
                    ant,
                    ant1,
                    ant2,
                )

    update = numpy.zeros((nants, nchan_gt, 2, 2), "complex")
    for ant in range(nants):
        for ch in range(nchan_gt):
            if numpy.linalg.matrix_rank(Smm[ant, ch]) != 2:
                # set a flag or zero a weight
                continue
            update[ant, ch] = Som[ant, ch] @ numpy.linalg.inv(Smm[ant, ch])

    return update


def _calc_and_solve_normal_equations(
    gX,
    gY,
    dXY,
    dYX,
    vis,
    modelvis,
    wgt,  # pylint: disable=unused-argument
    ant1,
    ant2,
):
    """
    Calculate and solve normal equations for linearised gain and leakage terms

    :param gX,gY,dXY,dYX: 2D numpy arrays containing the initial complex gain
        and leakage estimates [nant, nchan]
    :param vis: Visibility containing the observed data_models
    :param modelvis: Visibility containing the visibility predicted by a model
    :param wgt:
    :param ant1:
    :param ant2:
    :return [gX,gY,dXY,dYX]: 2D numpy arrays containing the complex gain and
        leakage updates [nant, nchan]
    """
    nants, nchan_gt = gX.shape
    ntime, nbl, nchan_vis, _ = vis.shape
    # number of output gain channels can be 1 in total or 1 per vis channel
    chgt = numpy.zeros(nchan_vis, "int") if nchan_gt == 1 else range(nchan_vis)

    vobs = vis.reshape(ntime, nbl, nchan_vis, 2, 2)
    vmdl = modelvis.reshape(ntime, nbl, nchan_vis, 2, 2)

    gX_update = numpy.zeros((nants, nchan_gt), "complex")
    gY_update = numpy.zeros((nants, nchan_gt), "complex")
    dXY_update = numpy.zeros((nants, nchan_gt), "complex")
    dYX_update = numpy.zeros((nants, nchan_gt), "complex")

    # accumulation space for normal eqautions products
    #  - all stations x 4 pol x real & imag
    AA = numpy.zeros([8 * nants, 8 * nants])
    Av = numpy.zeros([8 * nants, 1])

    for f in range(nchan_vis):
        ch = chgt[f]

        A = numpy.zeros([8 * nbl, 8 * nants])
        dv = numpy.zeros([8 * nbl, 1])

        for t in range(ntime):
            for k in range(nbl):
                if ant1[k] == ant2[k]:
                    continue

                i = ant1[k]
                j = ant2[k]

                ix2 = 2 * i
                jx2 = 2 * j
                kx2 = 2 * k

                # perhaps move most of these into update_design_matrix?

                # visibility indices (i.e. design matrix rows)
                kXX = kx2
                kYY = kx2 + 2 * nbl
                kXY = kx2 + 4 * nbl
                kYX = kx2 + 6 * nbl

                # parameter indices (i.e. design matrix columns)
                iXX = ix2
                iYY = ix2 + 2 * nants
                iXY = ix2 + 4 * nants
                iYX = ix2 + 6 * nants

                jXX = jx2
                jYY = jx2 + 2 * nants
                jXY = jx2 + 4 * nants
                jYX = jx2 + 6 * nants

                # Update the normal equation data vector for this visibility

                sres = vobs[t, k, f] - vmdl[t, k, f]

                dv[kXX] = numpy.real(sres[0, 0])
                dv[kYY] = numpy.real(sres[1, 1])
                dv[kXY] = numpy.real(sres[0, 1])
                dv[kYX] = numpy.real(sres[1, 0])

                dv[kXX + 1] = numpy.imag(sres[0, 0])
                dv[kYY + 1] = numpy.imag(sres[1, 1])
                dv[kXY + 1] = numpy.imag(sres[0, 1])
                dv[kYX + 1] = numpy.imag(sres[1, 0])

                # Update the normal equation design matrix with the first
                # derivatives of the real and imag parts of linearly polarised
                # visibilities for baseline i-j with respect to all relevant
                # gain and leakage free parameters.

                update_design_matrix(
                    A,
                    kXX,
                    kYY,
                    kXY,
                    kYX,
                    iXX,
                    iYY,
                    iXY,
                    iYX,
                    jXX,
                    jYY,
                    jXY,
                    jYX,
                    vmdl[t, k, f, 0, 0],
                    vmdl[t, k, f, 1, 1],
                    vmdl[t, k, f, 0, 1],
                    vmdl[t, k, f, 1, 0],
                    gX[i, ch],
                    gY[i, ch],
                    dXY[i, ch],
                    dYX[i, ch],
                    gX[j, ch],
                    gY[j, ch],
                    dXY[j, ch],
                    dYX[j, ch],
                )

        # update normal equations for this channel
        AA += A.T @ A
        Av += A.T @ dv

        # if each channel needs its own solution or this is the last channel,
        # solve the normal equations
        if nchan_gt == nchan_vis or f == nchan_vis - 1:
            gfit = lsmr(csc_matrix(AA, dtype=float), Av)[0]

            gX_update[:, ch] = (
                gfit[0 * nants : 2 * nants - 1 : 2]
                + 1j * gfit[0 * nants + 1 : 2 * nants : 2]
            )
            gY_update[:, ch] = (
                gfit[2 * nants : 4 * nants - 1 : 2]
                + 1j * gfit[2 * nants + 1 : 4 * nants : 2]
            )
            dXY_update[:, ch] = (
                gfit[4 * nants : 6 * nants - 1 : 2]
                + 1j * gfit[4 * nants + 1 : 6 * nants : 2]
            )
            dYX_update[:, ch] = (
                gfit[6 * nants : 8 * nants - 1 : 2]
                + 1j * gfit[6 * nants + 1 : 8 * nants : 2]
            )

            if f < nchan_vis - 1:
                # reset the accumulation arrays
                AA = numpy.zeros([8 * nants, 8 * nants])
                Av = numpy.zeros([8 * nants, 1])

    return [gX_update, gY_update, dXY_update, dYX_update]


def _calc_and_solve_normal_equations_with_presumming(
    gX,
    gY,
    dXY,
    dYX,
    Smo,
    Smm,
    ant1,
    ant2,
):
    """
    Calculate and solve normal equations for linearised gain and leakage terms

    :param gX,gY,dXY,dYX: 2D numpy arrays containing the initial complex gain
        and leakage estimates [nant]
    :param vis: Visibility containing the observed data_models
    :param modelvis: Visibility containing the visibility predicted by a model
    :param ant1:
    :param ant2:
    :return [gX,gY,dXY,dYX]: 2D numpy arrays containing the complex gain and
        leakage updates [nant]
    """
    nants = gX.shape[0]
    nbl = Smo.shape[0]

    gX_update = numpy.zeros(nants, "complex")
    gY_update = numpy.zeros(nants, "complex")
    dXY_update = numpy.zeros(nants, "complex")
    dYX_update = numpy.zeros(nants, "complex")

    # accumulation space for normal eqautions products
    #  - all stations x 4 pol x real & imag
    AA = numpy.zeros([8 * nants, 8 * nants])
    Av = numpy.zeros([8 * nants, 1])

    for k in range(nbl):
        if ant1[k] == ant2[k]:
            continue

        i = ant1[k]
        j = ant2[k]

        ix2 = 2 * i
        jx2 = 2 * j

        # parameter indices (i.e. row and column indices)
        iXX = ix2
        iYY = ix2 + 2 * nants
        iXY = ix2 + 4 * nants
        iYX = ix2 + 6 * nants

        jXX = jx2
        jYY = jx2 + 2 * nants
        jXY = jx2 + 4 * nants
        jYX = jx2 + 6 * nants

        # Generate a 4x4 Complex Diff matrix for each relevant gain
        # and leakage free parameter of baseline i-j.
        params = gen_cdm(
            gX[i], gY[i], dXY[i], dYX[i], gX[j], gY[j], dXY[j], dYX[j]
        )
        pRe = params[0:16:2]
        pIm = params[1:16:2]
        # Generate the 4x4 gain matrix for baseline i-j.
        values = gen_pol_matrix(
            gX[i], gY[i], dXY[i], dYX[i], gX[j], gY[j], dXY[j], dYX[j]
        )

        pos = numpy.zeros(8, "int")
        pos[0] = iXX
        pos[1] = jXX
        pos[2] = iYY
        pos[3] = jYY
        pos[4] = iXY
        pos[5] = jXY
        pos[6] = iYX
        pos[7] = jYX

        for param1 in range(8):
            pos1 = pos[param1]
            v_re = 0
            v_im = 0
            for p in range(0, 4):
                v_re += numpy.real(
                    numpy.conj(pRe[param1][p, :][numpy.newaxis, :])
                    @ (
                        Smo[k][:, p][:, numpy.newaxis]
                        - Smm[k] @ values[p, :][numpy.newaxis, :].T
                    )
                )
                v_im += numpy.real(
                    numpy.conj(pIm[param1][p, :][numpy.newaxis, :])
                    @ (
                        Smo[k][:, p][:, numpy.newaxis]
                        - Smm[k] @ values[p, :][numpy.newaxis, :].T
                    )
                )
            Av[pos[param1]] += v_re[0, 0]
            Av[pos[param1] + 1] += v_im[0, 0]
            p1ReH = pRe[param1].conj().T
            p1ImH = pIm[param1].conj().T
            for param2 in range(8):
                pos2 = pos[param2]
                AA[pos1, pos2] += numpy.sum(
                    numpy.real((p1ReH @ pRe[param2]) * Smm[k])
                )
                AA[pos1, pos2 + 1] += numpy.sum(
                    numpy.real((p1ReH @ pIm[param2]) * Smm[k])
                )
                AA[pos1 + 1, pos2] += numpy.sum(
                    numpy.real((p1ImH @ pRe[param2]) * Smm[k])
                )
                AA[pos1 + 1, pos2 + 1] += numpy.sum(
                    numpy.real((p1ImH @ pIm[param2]) * Smm[k])
                )

    gfit = lsmr(csc_matrix(AA, dtype=float), Av)[0]

    gX_update = (
        gfit[0 * nants : 2 * nants - 1 : 2]
        + 1j * gfit[0 * nants + 1 : 2 * nants : 2]
    )
    gY_update = (
        gfit[2 * nants : 4 * nants - 1 : 2]
        + 1j * gfit[2 * nants + 1 : 4 * nants : 2]
    )
    dXY_update = (
        gfit[4 * nants : 6 * nants - 1 : 2]
        + 1j * gfit[4 * nants + 1 : 6 * nants : 2]
    )
    dYX_update = (
        gfit[6 * nants : 8 * nants - 1 : 2]
        + 1j * gfit[6 * nants + 1 : 8 * nants : 2]
    )

    return [gX_update, gY_update, dXY_update, dYX_update]


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
