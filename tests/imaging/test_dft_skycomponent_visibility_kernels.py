"""
Unit tests for DFT kernels
"""
import numpy
import pytest

pytest.importorskip(
    modname="ska_sdp_func", reason="ska-sdp-func is an optional dependency"
)
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility


@pytest.mark.parametrize(
    "compute_kernel", ["cpu_looped", "gpu_cupy_raw", "proc_func"]
)
def test_dft_stokesiquv_visibility(compute_kernel, visibility, comp_dft):
    """
    The various DFT kernels return the same results
    """
    if compute_kernel == "gpu_cupy_raw":
        try:
            # pylint: disable=unused-import,import-outside-toplevel
            import cupy  # noqa: F401
        except ModuleNotFoundError:
            return

    new_vis = visibility.copy(deep=True)
    result = dft_skycomponent_visibility(
        new_vis,
        comp_dft,
        dft_compute_kernel=compute_kernel,
    )
    qa = result.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(qa.data["maxabs"], 2400.0)
    numpy.testing.assert_almost_equal(qa.data["minabs"], 200.9975124)
    numpy.testing.assert_almost_equal(qa.data["rms"], 942.9223125)

    numpy.testing.assert_almost_equal(
        result.vis.data.sum(),
        15767919.209378432 + 777296.8338094898j,
        decimal=4,
    )
