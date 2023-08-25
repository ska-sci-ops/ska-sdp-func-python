"""
Unit tests for chain calibration functions
"""
import logging

import numpy
import pytest
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)

pytest.importorskip(
    modname="ska_sdp_func", reason="ska-sdp-func is an optional dependency"
)
from ska_sdp_func_python.calibration.chain_calibration import (
    apply_calibration_chain,
    calibrate_chain,
    create_calibration_controls,
    solve_calibrate_chain,
)
from ska_sdp_func_python.calibration.operations import apply_gaintable
from tests.testing_utils import simulate_gaintable, vis_with_component_data

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.mark.parametrize("context", ["T", "TG", "B", "GB"])
def test_apply_calibration_chain(context):
    """
    Unit test for apply_calibration_chain
    """

    vis = vis_with_component_data("stokesI", "stokesI", [100.0])
    original = vis.copy(deep=True)
    gain_table = create_gaintable_from_visibility(vis)
    gain_table = simulate_gaintable(
        gain_table,
        phase_error=1.0,
        amplitude_error=0.0,
        leakage=0.0,
    )

    controls = create_calibration_controls()

    new_vis = apply_calibration_chain(
        vis, gain_table, calibration_context=context, controls=controls
    )

    # Visibility is changed only when context is "T"
    if context in ["T", "TG"]:
        assert not numpy.array_equal(original["vis"].data, new_vis["vis"].data)
        err = numpy.max(
            numpy.abs(
                original.visibility_acc.flagged_vis
                - new_vis.visibility_acc.flagged_vis
            )
        )
        assert err < 200, err
    else:
        assert numpy.array_equal(original["vis"].data, new_vis["vis"].data)


@pytest.mark.parametrize(
    "context, phase_only, first_selfcal_value",
    [
        ("T", False, 0.0),
        ("T", True, 0.0),
        ("G", False, 0.0),
        ("TG", False, 0.0),
        ("B", False, 10.0),
    ],
)
def test_calibrate_chain(context, phase_only, first_selfcal_value):
    """
    Test calibrate_chain for different calibration contexts
    e.g. ("T", "G", "TG", "B")
    """

    vis = vis_with_component_data("stokesI", "stokesI", [100.0])
    gain_table = create_gaintable_from_visibility(vis)
    gain_table = simulate_gaintable(
        gain_table,
        phase_error=10.0,
        amplitude_error=0.0,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gain_table)

    controls = create_calibration_controls()
    if context in ["T", "G"]:
        controls[context]["first_selfcal"] = first_selfcal_value
        controls[context]["phase_only"] = phase_only
        controls[context]["timeslice"] = 0.0
        gaintable_var = context

    elif context == "TG":
        controls["T"]["first_selfcal"] = first_selfcal_value
        controls["T"]["timeslice"] = 0.0
        controls["T"]["phase_only"] = True
        controls["G"]["first_selfcal"] = first_selfcal_value
        controls["G"]["timeslice"] = 0.0
        controls["G"]["phase_only"] = phase_only
        gaintable_var = "G"

    elif context == "B":
        controls["T"]["first_selfcal"] = first_selfcal_value
        controls["T"]["timeslice"] = 0.0
        controls["T"]["phase_only"] = True

        controls["G"]["first_selfcal"] = first_selfcal_value
        controls["G"]["timeslice"] = 0.0
        controls["G"]["phase_only"] = True

        controls["B"]["first_selfcal"] = 0
        controls["B"]["timeslice"] = 0.0
        controls["B"]["phase_only"] = phase_only
        gaintable_var = context

    _, gt_dict = calibrate_chain(
        vis, original, calibration_context=context, controls=controls
    )
    residual = numpy.max(gt_dict[gaintable_var].residual)
    assert residual < 1.3e-6, ("Max residual = %s", residual)


@pytest.mark.parametrize("context", ["T", "TG", "B"])
def test_solve_calibration_chain(context):
    """
    Unit test for solve_calibration_chain
    """
    vis = vis_with_component_data("stokesI", "stokesI", [100.0])
    gain_table = create_gaintable_from_visibility(vis)
    gain_table = simulate_gaintable(
        gain_table,
        phase_error=10.0,
        amplitude_error=0.0,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gain_table)

    controls = create_calibration_controls()
    gt_dict = solve_calibrate_chain(
        vis, original, calibration_context=context, controls=controls
    )
    if context == "TG":
        gaintable_var = "T"
    else:
        gaintable_var = context

    residual = numpy.max(gt_dict[gaintable_var].residual)
    assert residual < 1.3e-6, ("Max residual = %s", residual)
