""" Utility functions used by solvers. Based on the Yandasoft algorithm. """

import numpy as np
from numpy import conj as conj
from numpy import imag as imag
from numpy import real as real


def update_design_matrix(
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
    XX,
    YY,
    XY,
    YX,
    gXi,
    gYi,
    dXYi,
    dYXi,
    gXj,
    gYj,
    dXYj,
    dYXj,
):
    """
    Update the normal equation design matrix with the first derivatives of the
    real and imag parts of linearly polarised visibilities for baseline i-j
    with respect to all relevant gain and leakage free parameters.

    :param A: Normal equation design matrix to update. This matrix has
        dimensions [8 x num_visibility, 8 x num_antenna], with the factors of
        8 coming from 4 linear polarisations multiplied by the two complex
        components (stored sequentially as real,imag).
    :param k??: row index for the real part this visibility for polarisation ??
    :param i??: column index for the real part the first antenna in the
        baseline for polarisation ??
    :param j??: column index for the real part the second antenna in the
        baseline for polarisation ??
    :param XX,XY,YX,YY: complex model visibilities
    :param gXi,gYi,dXYi,dYXi: current estimates of the complex gains and
        leakage terms for the first antenna in the baseline
    :param gXj,gYj,dXYj,dYXj: current estimates of the complex gains and
        leakage terms for the second antenna in the baseline
    """

    # {d_V}{d_real(gXi)}

    tmpXX = (
        +conj(gXj) * XX
        + conj(gXj * dXYj) * XY
        + dXYi * conj(gXj) * YX
        + dXYi * conj(gXj * dXYj) * YY
    )
    tmpXY = (
        -conj(gYj * dYXj) * XX
        + conj(gYj) * XY
        - dXYi * conj(gYj * dYXj) * YX
        + dXYi * conj(gYj) * YY
    )
    tmpYX = +0 * XX + 0 * YX + 0 * YX + 0 * YY
    tmpYY = +0 * XX + 0 * XY + 0 * YX + 0 * YY

    A[kXX, iXX] += +real(tmpXX)
    A[kXY, iXX] += +real(tmpXY)
    A[kYX, iXX] += +real(tmpYX)
    A[kYY, iXX] += +real(tmpYY)

    A[kXX + 1, iXX] += +imag(tmpXX)
    A[kXY + 1, iXX] += +imag(tmpXY)
    A[kYX + 1, iXX] += +imag(tmpYX)
    A[kYY + 1, iXX] += +imag(tmpYY)

    # {d_V}{d_imag(gXi)}

    tmpXX = (
        +conj(gXj) * XX
        + conj(gXj * dXYj) * XY
        + dXYi * conj(gXj) * YX
        + dXYi * conj(gXj * dXYj) * YY
    )
    tmpXY = (
        -conj(gYj * dYXj) * XX
        + conj(gYj) * XY
        - dXYi * conj(gYj * dYXj) * YX
        + dXYi * conj(gYj) * YY
    )
    tmpYX = +0 * XX + 0 * YX + 0 * YX + 0 * YY
    tmpYY = +0 * XX + 0 * XY + 0 * YX + 0 * YY

    A[kXX, iXX + 1] += -imag(tmpXX)  # imag sign:
    A[kXY, iXX + 1] += -imag(tmpXY)
    A[kYX, iXX + 1] += -imag(tmpYX)
    A[kYY, iXX + 1] += -imag(tmpYY)

    A[kXX + 1, iXX + 1] += +real(tmpXX)  # imag sign:
    A[kXY + 1, iXX + 1] += +real(tmpXY)
    A[kYX + 1, iXX + 1] += +real(tmpYX)
    A[kYY + 1, iXX + 1] += +real(tmpYY)

    # {d_V}{d_real(gXj)}

    tmpXX = (
        +gXi * XX
        + gXi * conj(dXYj) * XY
        + gXi * dXYi * YX
        + gXi * dXYi * conj(dXYj) * YY
    )
    tmpXY = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpYX = (
        -gYi * dYXi * XX
        - gYi * dYXi * conj(dXYj) * YX
        + gYi * YX
        + gYi * conj(dXYj) * YY
    )
    tmpYY = +0 * XX + 0 * XY + 0 * YX + 0 * YY

    A[kXX, jXX] += +real(tmpXX)
    A[kXY, jXX] += +real(tmpXY)
    A[kYX, jXX] += +real(tmpYX)
    A[kYY, jXX] += +real(tmpYY)

    A[kXX + 1, jXX] += +imag(tmpXX)
    A[kXY + 1, jXX] += +imag(tmpXY)
    A[kYX + 1, jXX] += +imag(tmpYX)
    A[kYY + 1, jXX] += +imag(tmpYY)

    # {d_V}{d_imag(gXj)}

    tmpXX = (
        +gXi * XX
        + gXi * conj(dXYj) * XY
        + gXi * dXYi * YX
        + gXi * dXYi * conj(dXYj) * YY
    )
    tmpXY = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpYX = (
        -gYi * dYXi * XX
        - gYi * dYXi * conj(dXYj) * YX
        + gYi * YX
        + gYi * conj(dXYj) * YY
    )
    tmpYY = +0 * XX + 0 * XY + 0 * YX + 0 * YY

    A[kXX, jXX + 1] += +imag(tmpXX)  # imag sign:
    A[kXY, jXX + 1] += +imag(tmpXY)
    A[kYX, jXX + 1] += +imag(tmpYX)
    A[kYY, jXX + 1] += +imag(tmpYY)

    A[kXX + 1, jXX + 1] += -real(tmpXX)  # imag sign:
    A[kXY + 1, jXX + 1] += -real(tmpXY)
    A[kYX + 1, jXX + 1] += -real(tmpYX)
    A[kYY + 1, jXX + 1] += -real(tmpYY)

    # {d_V}{d_real(gYi)}

    tmpXX = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpXY = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpYX = (
        -dYXi * conj(gXj) * XX
        - dYXi * conj(gXj * dXYj) * YX
        + conj(gXj) * YX
        + conj(gXj * dXYj) * YY
    )
    tmpYY = (
        +dYXi * conj(gYj * dYXj) * XX
        - dYXi * conj(gYj) * XY
        - conj(gYj * dYXj) * YX
        + conj(gYj) * YY
    )

    A[kXX, iYY] += +real(tmpXX)
    A[kXY, iYY] += +real(tmpXY)
    A[kYX, iYY] += +real(tmpYX)
    A[kYY, iYY] += +real(tmpYY)

    A[kXX + 1, iYY] += +imag(tmpXX)
    A[kXY + 1, iYY] += +imag(tmpXY)
    A[kYX + 1, iYY] += +imag(tmpYX)
    A[kYY + 1, iYY] += +imag(tmpYY)

    # {d_V}{d_imag(gYi)}

    tmpXX = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpXY = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpYX = (
        -dYXi * conj(gXj) * XX
        - dYXi * conj(gXj * dXYj) * YX
        + conj(gXj) * YX
        + conj(gXj * dXYj) * YY
    )
    tmpYY = (
        +dYXi * conj(gYj * dYXj) * XX
        - dYXi * conj(gYj) * XY
        - conj(gYj * dYXj) * YX
        + conj(gYj) * YY
    )

    A[kXX, iYY + 1] += -imag(tmpXX)  # imag sign:
    A[kXY, iYY + 1] += -imag(tmpXY)
    A[kYX, iYY + 1] += -imag(tmpYX)
    A[kYY, iYY + 1] += -imag(tmpYY)

    A[kXX + 1, iYY + 1] += +real(tmpXX)  # imag sign:
    A[kXY + 1, iYY + 1] += +real(tmpXY)
    A[kYX + 1, iYY + 1] += +real(tmpYX)
    A[kYY + 1, iYY + 1] += +real(tmpYY)

    # {d_V}{d_real(gYj)}

    tmpXX = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpXY = (
        -gXi * conj(dYXj) * XX
        + gXi * XY
        - gXi * dXYi * conj(dYXj) * YX
        + gXi * dXYi * YY
    )
    tmpYX = +0 * XX + 0 * YX + 0 * YX + 0 * YY
    tmpYY = (
        +gYi * dYXi * conj(dYXj) * XX
        - gYi * dYXi * XY
        - gYi * conj(dYXj) * YX
        + gYi * YY
    )

    A[kXX, jYY] += +real(tmpXX)
    A[kXY, jYY] += +real(tmpXY)
    A[kYX, jYY] += +real(tmpYX)
    A[kYY, jYY] += +real(tmpYY)

    A[kXX + 1, jYY] += +imag(tmpXX)
    A[kXY + 1, jYY] += +imag(tmpXY)
    A[kYX + 1, jYY] += +imag(tmpYX)
    A[kYY + 1, jYY] += +imag(tmpYY)

    # {d_V}{d_imag(gYj)}

    tmpXX = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpXY = (
        -gXi * conj(dYXj) * XX
        + gXi * XY
        - gXi * dXYi * conj(dYXj) * YX
        + gXi * dXYi * YY
    )
    tmpYX = +0 * XX + 0 * YX + 0 * YX + 0 * YY
    tmpYY = (
        +gYi * dYXi * conj(dYXj) * XX
        - gYi * dYXi * XY
        - gYi * conj(dYXj) * YX
        + gYi * YY
    )

    A[kXX, jYY + 1] += +imag(tmpXX)  # imag sign:
    A[kXY, jYY + 1] += +imag(tmpXY)
    A[kYX, jYY + 1] += +imag(tmpYX)
    A[kYY, jYY + 1] += +imag(tmpYY)

    A[kXX + 1, jYY + 1] += -real(tmpXX)  # imag sign:
    A[kXY + 1, jYY + 1] += -real(tmpXY)
    A[kYX + 1, jYY + 1] += -real(tmpYX)
    A[kYY + 1, jYY + 1] += -real(tmpYY)

    # {d_V}{d_real(dXYi)}

    tmpXX = (
        +0 * XX + 0 * XY + gXi * conj(gXj) * YX + gXi * conj(gXj * dXYj) * YY
    )
    tmpXY = (
        +0 * XX + 0 * XY - gXi * conj(gYj * dYXj) * YX + gXi * conj(gYj) * YY
    )
    tmpYX = +0 * XX + 0 * YX + 0 * YX + 0 * YY
    tmpYY = +0 * XX + 0 * XY + 0 * YX + 0 * YY

    A[kXX, iXY] += +real(tmpXX)
    A[kXY, iXY] += +real(tmpXY)
    A[kYX, iXY] += +real(tmpYX)
    A[kYY, iXY] += +real(tmpYY)

    A[kXX + 1, iXY] += +imag(tmpXX)
    A[kXY + 1, iXY] += +imag(tmpXY)
    A[kYX + 1, iXY] += +imag(tmpYX)
    A[kYY + 1, iXY] += +imag(tmpYY)

    # {d_V}{d_imag(dXYi)}

    tmpXX = (
        +0 * XX + 0 * XY + gXi * conj(gXj) * YX + gXi * conj(gXj * dXYj) * YY
    )
    tmpXY = (
        +0 * XX + 0 * XY - gXi * conj(gYj * dYXj) * YX + gXi * conj(gYj) * YY
    )
    tmpYX = +0 * XX + 0 * YX + 0 * YX + 0 * YY
    tmpYY = +0 * XX + 0 * XY + 0 * YX + 0 * YY

    A[kXX, iXY + 1] += -imag(tmpXX)  # imag sign:
    A[kXY, iXY + 1] += -imag(tmpXY)
    A[kYX, iXY + 1] += -imag(tmpYX)
    A[kYY, iXY + 1] += -imag(tmpYY)

    A[kXX + 1, iXY + 1] += +real(tmpXX)  # imag sign:
    A[kXY + 1, iXY + 1] += +real(tmpXY)
    A[kYX + 1, iXY + 1] += +real(tmpYX)
    A[kYY + 1, iXY + 1] += +real(tmpYY)

    # {d_V}{d_real(dXYj)}

    tmpXX = (
        +0 * XX + gXi * conj(gXj) * XY + 0 * YX + gXi * dXYi * conj(gXj) * YY
    )
    tmpXY = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpYX = (
        +0 * XX - gYi * dYXi * conj(gXj) * YX + 0 * YX + gYi * conj(gXj) * YY
    )
    tmpYY = +0 * XX + 0 * XY + 0 * YX + 0 * YY

    A[kXX, jXY] += +real(tmpXX)
    A[kXY, jXY] += +real(tmpXY)
    A[kYX, jXY] += +real(tmpYX)
    A[kYY, jXY] += +real(tmpYY)

    A[kXX + 1, jXY] += +imag(tmpXX)
    A[kXY + 1, jXY] += +imag(tmpXY)
    A[kYX + 1, jXY] += +imag(tmpYX)
    A[kYY + 1, jXY] += +imag(tmpYY)

    # {d_V}{d_imag(dXYj)}

    tmpXX = (
        +0 * XX + gXi * conj(gXj) * XY + 0 * YX + gXi * dXYi * conj(gXj) * YY
    )
    tmpXY = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpYX = (
        +0 * XX - gYi * dYXi * conj(gXj) * YX + 0 * YX + gYi * conj(gXj) * YY
    )
    tmpYY = +0 * XX + 0 * XY + 0 * YX + 0 * YY

    A[kXX, jXY + 1] += +imag(tmpXX)  # imag sign:
    A[kXY, jXY + 1] += +imag(tmpXY)
    A[kYX, jXY + 1] += +imag(tmpYX)
    A[kYY, jXY + 1] += +imag(tmpYY)

    A[kXX + 1, jXY + 1] += -real(tmpXX)  # imag sign:
    A[kXY + 1, jXY + 1] += -real(tmpXY)
    A[kYX + 1, jXY + 1] += -real(tmpYX)
    A[kYY + 1, jXY + 1] += -real(tmpYY)

    # {d_V}{d_real(dYXi)}

    tmpXX = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpXY = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpYX = (
        -gYi * conj(gXj) * XX - gYi * conj(gXj * dXYj) * YX + 0 * YX + 0 * YY
    )
    tmpYY = (
        +gYi * conj(gYj * dYXj) * XX - gYi * conj(gYj) * XY + 0 * YX + 0 * YY
    )

    A[kXX, iYX] += +real(tmpXX)
    A[kXY, iYX] += +real(tmpXY)
    A[kYX, iYX] += +real(tmpYX)
    A[kYY, iYX] += +real(tmpYY)

    A[kXX + 1, iYX] += +imag(tmpXX)
    A[kXY + 1, iYX] += +imag(tmpXY)
    A[kYX + 1, iYX] += +imag(tmpYX)
    A[kYY + 1, iYX] += +imag(tmpYY)

    # {d_V}{d_imag(dYXi)}

    tmpXX = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpXY = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpYX = (
        -gYi * conj(gXj) * XX - gYi * conj(gXj * dXYj) * YX + 0 * YX + 0 * YY
    )
    tmpYY = (
        +gYi * conj(gYj * dYXj) * XX - gYi * conj(gYj) * XY + 0 * YX + 0 * YY
    )

    A[kXX, iYX + 1] += -imag(tmpXX)  # imag sign:
    A[kXY, iYX + 1] += -imag(tmpXY)
    A[kYX, iYX + 1] += -imag(tmpYX)
    A[kYY, iYX + 1] += -imag(tmpYY)

    A[kXX + 1, iYX + 1] += +real(tmpXX)  # imag sign:
    A[kXY + 1, iYX + 1] += +real(tmpXY)
    A[kYX + 1, iYX + 1] += +real(tmpYX)
    A[kYY + 1, iYX + 1] += +real(tmpYY)

    # {d_V}{d_real(dYXj)}

    tmpXX = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpXY = (
        -gXi * conj(gYj) * XX + 0 * XY - gXi * dXYi * conj(gYj) * YX + 0 * YY
    )
    tmpYX = +0 * XX + 0 * YX + 0 * YX + 0 * YY
    tmpYY = (
        +gYi * dYXi * conj(gYj) * XX + 0 * XY - gYi * conj(gYj) * YX + 0 * YY
    )

    A[kXX, jYX] += +real(tmpXX)
    A[kXY, jYX] += +real(tmpXY)
    A[kYX, jYX] += +real(tmpYX)
    A[kYY, jYX] += +real(tmpYY)

    A[kXX + 1, jYX] += +imag(tmpXX)
    A[kXY + 1, jYX] += +imag(tmpXY)
    A[kYX + 1, jYX] += +imag(tmpYX)
    A[kYY + 1, jYX] += +imag(tmpYY)

    # {d_V}{d_imag(dYXj)}

    tmpXX = +0 * XX + 0 * XY + 0 * YX + 0 * YY
    tmpXY = (
        -gXi * conj(gYj) * XX + 0 * XY - gXi * dXYi * conj(gYj) * YX + 0 * YY
    )
    tmpYX = +0 * XX + 0 * YX + 0 * YX + 0 * YY
    tmpYY = (
        +gYi * dYXi * conj(gYj) * XX + 0 * XY - gYi * conj(gYj) * YX + 0 * YY
    )

    A[kXX, jYX + 1] += +imag(tmpXX)  # imag sign:
    A[kXY, jYX + 1] += +imag(tmpXY)
    A[kYX, jYX + 1] += +imag(tmpYX)
    A[kYY, jYX + 1] += +imag(tmpYY)

    A[kXX + 1, jYX + 1] += -real(tmpXX)  # imag sign:
    A[kXY + 1, jYX + 1] += -real(tmpXY)
    A[kYX + 1, jYX + 1] += -real(tmpYX)
    A[kYY + 1, jYX + 1] += -real(tmpYY)


def gen_cdm(gXi, gYi, dXYi, dYXi, gXj, gYj, dXYj, dYXj):
    """
    Generate a 4x4 Complex Diff matrix for each relevant gain and leakage free
    parameter of baseline i-j.

    Note: has XX XY YX YY ordering

    :param gXi,gYi,dXYi,dYXi: current estimates of the complex gain and leakage
        terms for the first antenna in the baseline
    :param gXj,gYj,dXYj,dYXj: current estimates of the complex gain and leakage
        terms for the second antenna in the baseline
    :return: List containing the 16 derivative 4x4 matrices
    """

    dfdgXiRe = real(
        np.array(
            [
                [
                    +conj(gXj),
                    +conj(gXj * dXYj),
                    +dXYi * conj(gXj),
                    +dXYi * conj(gXj * dXYj),
                ],
                [
                    -conj(gYj * dYXj),
                    +conj(gYj),
                    -dXYi * conj(gYj * dYXj),
                    +dXYi * conj(gYj),
                ],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    ) + 1j * imag(
        np.array(
            [
                [
                    +conj(gXj),
                    +conj(gXj * dXYj),
                    +dXYi * conj(gXj),
                    +dXYi * conj(gXj * dXYj),
                ],
                [
                    -conj(gYj * dYXj),
                    +conj(gYj),
                    -dXYi * conj(gYj * dYXj),
                    +dXYi * conj(gYj),
                ],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    )

    dfdgXiIm = -imag(
        np.array(
            [
                [
                    +conj(gXj),
                    +conj(gXj * dXYj),
                    +dXYi * conj(gXj),
                    +dXYi * conj(gXj * dXYj),
                ],
                [
                    -conj(gYj * dYXj),
                    +conj(gYj),
                    -dXYi * conj(gYj * dYXj),
                    +dXYi * conj(gYj),
                ],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    ) + 1j * real(
        np.array(
            [
                [
                    +conj(gXj),
                    +conj(gXj * dXYj),
                    +dXYi * conj(gXj),
                    +dXYi * conj(gXj * dXYj),
                ],
                [
                    -conj(gYj * dYXj),
                    +conj(gYj),
                    -dXYi * conj(gYj * dYXj),
                    +dXYi * conj(gYj),
                ],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    )

    dfdgXjRe = real(
        np.array(
            [
                [
                    +gXi,
                    +gXi * conj(dXYj),
                    +gXi * dXYi,
                    +gXi * dXYi * conj(dXYj),
                ],
                [0, 0, 0, 0],
                [
                    -gYi * dYXi,
                    -gYi * dYXi * conj(dXYj),
                    +gYi,
                    +gYi * conj(dXYj),
                ],
                [0, 0, 0, 0],
            ]
        )
    ) + 1j * imag(
        np.array(
            [
                [
                    +gXi,
                    +gXi * conj(dXYj),
                    +gXi * dXYi,
                    +gXi * dXYi * conj(dXYj),
                ],
                [0, 0, 0, 0],
                [
                    -gYi * dYXi,
                    -gYi * dYXi * conj(dXYj),
                    +gYi,
                    +gYi * conj(dXYj),
                ],
                [0, 0, 0, 0],
            ]
        )
    )

    dfdgXjIm = imag(
        np.array(
            [
                [
                    +gXi,
                    +gXi * conj(dXYj),
                    +gXi * dXYi,
                    +gXi * dXYi * conj(dXYj),
                ],
                [0, 0, 0, 0],
                [
                    -gYi * dYXi,
                    -gYi * dYXi * conj(dXYj),
                    +gYi,
                    +gYi * conj(dXYj),
                ],
                [0, 0, 0, 0],
            ]
        )
    ) - 1j * real(
        np.array(
            [
                [
                    +gXi,
                    +gXi * conj(dXYj),
                    +gXi * dXYi,
                    +gXi * dXYi * conj(dXYj),
                ],
                [0, 0, 0, 0],
                [
                    -gYi * dYXi,
                    -gYi * dYXi * conj(dXYj),
                    +gYi,
                    +gYi * conj(dXYj),
                ],
                [0, 0, 0, 0],
            ]
        )
    )

    dfdgYiRe = real(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [
                    -dYXi * conj(gXj),
                    -dYXi * conj(gXj * dXYj),
                    +conj(gXj),
                    +conj(gXj * dXYj),
                ],
                [
                    +dYXi * conj(gYj * dYXj),
                    -dYXi * conj(gYj),
                    -conj(gYj * dYXj),
                    +conj(gYj),
                ],
            ]
        )
    ) + 1j * imag(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [
                    -dYXi * conj(gXj),
                    -dYXi * conj(gXj * dXYj),
                    +conj(gXj),
                    +conj(gXj * dXYj),
                ],
                [
                    +dYXi * conj(gYj * dYXj),
                    -dYXi * conj(gYj),
                    -conj(gYj * dYXj),
                    +conj(gYj),
                ],
            ]
        )
    )

    dfdgYiIm = -imag(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [
                    -dYXi * conj(gXj),
                    -dYXi * conj(gXj * dXYj),
                    +conj(gXj),
                    +conj(gXj * dXYj),
                ],
                [
                    +dYXi * conj(gYj * dYXj),
                    -dYXi * conj(gYj),
                    -conj(gYj * dYXj),
                    +conj(gYj),
                ],
            ]
        )
    ) + 1j * real(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [
                    -dYXi * conj(gXj),
                    -dYXi * conj(gXj * dXYj),
                    +conj(gXj),
                    +conj(gXj * dXYj),
                ],
                [
                    +dYXi * conj(gYj * dYXj),
                    -dYXi * conj(gYj),
                    -conj(gYj * dYXj),
                    +conj(gYj),
                ],
            ]
        )
    )

    dfdgYjRe = real(
        np.array(
            [
                [0, 0, 0, 0],
                [
                    -gXi * conj(dYXj),
                    +gXi,
                    -gXi * dXYi * conj(dYXj),
                    +gXi * dXYi,
                ],
                [0, 0, 0, 0],
                [
                    +gYi * dYXi * conj(dYXj),
                    -gYi * dYXi,
                    -gYi * conj(dYXj),
                    +gYi,
                ],
            ]
        )
    ) + 1j * imag(
        np.array(
            [
                [0, 0, 0, 0],
                [
                    -gXi * conj(dYXj),
                    +gXi,
                    -gXi * dXYi * conj(dYXj),
                    +gXi * dXYi,
                ],
                [0, 0, 0, 0],
                [
                    +gYi * dYXi * conj(dYXj),
                    -gYi * dYXi,
                    -gYi * conj(dYXj),
                    +gYi,
                ],
            ]
        )
    )

    dfdgYjIm = imag(
        np.array(
            [
                [0, 0, 0, 0],
                [
                    -gXi * conj(dYXj),
                    +gXi,
                    -gXi * dXYi * conj(dYXj),
                    +gXi * dXYi,
                ],
                [0, 0, 0, 0],
                [
                    +gYi * dYXi * conj(dYXj),
                    -gYi * dYXi,
                    -gYi * conj(dYXj),
                    +gYi,
                ],
            ]
        )
    ) - 1j * real(
        np.array(
            [
                [0, 0, 0, 0],
                [
                    -gXi * conj(dYXj),
                    +gXi,
                    -gXi * dXYi * conj(dYXj),
                    +gXi * dXYi,
                ],
                [0, 0, 0, 0],
                [
                    +gYi * dYXi * conj(dYXj),
                    -gYi * dYXi,
                    -gYi * conj(dYXj),
                    +gYi,
                ],
            ]
        )
    )

    dfddXYiRe = real(
        np.array(
            [
                [0, 0, +gXi * conj(gXj), +gXi * conj(gXj * dXYj)],
                [0, 0, -gXi * conj(gYj * dYXj), +gXi * conj(gYj)],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    ) + 1j * imag(
        np.array(
            [
                [0, 0, +gXi * conj(gXj), +gXi * conj(gXj * dXYj)],
                [0, 0, -gXi * conj(gYj * dYXj), +gXi * conj(gYj)],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    )

    dfddXYiIm = -imag(
        np.array(
            [
                [0, 0, +gXi * conj(gXj), +gXi * conj(gXj * dXYj)],
                [0, 0, -gXi * conj(gYj * dYXj), +gXi * conj(gYj)],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    ) + 1j * real(
        np.array(
            [
                [0, 0, +gXi * conj(gXj), +gXi * conj(gXj * dXYj)],
                [0, 0, -gXi * conj(gYj * dYXj), +gXi * conj(gYj)],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    )

    dfddXYjRe = real(
        np.array(
            [
                [0, +gXi * conj(gXj), 0, +gXi * dXYi * conj(gXj)],
                [0, 0, 0, 0],
                [0, -gYi * dYXi * conj(gXj), 0, +gYi * conj(gXj)],
                [0, 0, 0, 0],
            ]
        )
    ) + 1j * imag(
        np.array(
            [
                [0, +gXi * conj(gXj), 0, +gXi * dXYi * conj(gXj)],
                [0, 0, 0, 0],
                [0, -gYi * dYXi * conj(gXj), 0, +gYi * conj(gXj)],
                [0, 0, 0, 0],
            ]
        )
    )

    dfddXYjIm = imag(
        np.array(
            [
                [0, +gXi * conj(gXj), 0, +gXi * dXYi * conj(gXj)],
                [0, 0, 0, 0],
                [0, -gYi * dYXi * conj(gXj), 0, +gYi * conj(gXj)],
                [0, 0, 0, 0],
            ]
        )
    ) - 1j * real(
        np.array(
            [
                [0, +gXi * conj(gXj), 0, +gXi * dXYi * conj(gXj)],
                [0, 0, 0, 0],
                [0, -gYi * dYXi * conj(gXj), 0, +gYi * conj(gXj)],
                [0, 0, 0, 0],
            ]
        )
    )

    dfddYXiRe = real(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-gYi * conj(gXj), -gYi * conj(gXj * dXYj), 0, 0],
                [+gYi * conj(gYj * dYXj), -gYi * conj(gYj), 0, 0],
            ]
        )
    ) + 1j * imag(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-gYi * conj(gXj), -gYi * conj(gXj * dXYj), 0, 0],
                [+gYi * conj(gYj * dYXj), -gYi * conj(gYj), 0, 0],
            ]
        )
    )

    dfddYXiIm = -imag(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-gYi * conj(gXj), -gYi * conj(gXj * dXYj), 0, 0],
                [+gYi * conj(gYj * dYXj), -gYi * conj(gYj), 0, 0],
            ]
        )
    ) + 1j * real(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-gYi * conj(gXj), -gYi * conj(gXj * dXYj), 0, 0],
                [+gYi * conj(gYj * dYXj), -gYi * conj(gYj), 0, 0],
            ]
        )
    )

    dfddYXjRe = real(
        np.array(
            [
                [0, 0, 0, 0],
                [-gXi * conj(gYj), 0, -gXi * dXYi * conj(gYj), 0],
                [0, 0, 0, 0],
                [+gYi * dYXi * conj(gYj), 0, -gYi * conj(gYj), 0],
            ]
        )
    ) + 1j * imag(
        np.array(
            [
                [0, 0, 0, 0],
                [-gXi * conj(gYj), 0, -gXi * dXYi * conj(gYj), 0],
                [0, 0, 0, 0],
                [+gYi * dYXi * conj(gYj), 0, -gYi * conj(gYj), 0],
            ]
        )
    )

    dfddYXjIm = imag(
        np.array(
            [
                [0, 0, 0, 0],
                [-gXi * conj(gYj), 0, -gXi * dXYi * conj(gYj), 0],
                [0, 0, 0, 0],
                [+gYi * dYXi * conj(gYj), 0, -gYi * conj(gYj), 0],
            ]
        )
    ) - 1j * real(
        np.array(
            [
                [0, 0, 0, 0],
                [-gXi * conj(gYj), 0, -gXi * dXYi * conj(gYj), 0],
                [0, 0, 0, 0],
                [+gYi * dYXi * conj(gYj), 0, -gYi * conj(gYj), 0],
            ]
        )
    )

    return [
        dfdgXiRe,
        dfdgXiIm,
        dfdgXjRe,
        dfdgXjIm,
        dfdgYiRe,
        dfdgYiIm,
        dfdgYjRe,
        dfdgYjIm,
        dfddXYiRe,
        dfddXYiIm,
        dfddXYjRe,
        dfddXYjIm,
        dfddYXiRe,
        dfddYXiIm,
        dfddYXjRe,
        dfddYXjIm,
    ]


def gen_pol_matrix(gXi, gYi, dXYi, dYXi, gXj, gYj, dXYj, dYXj):
    """
    Generate the 4x4 gain matrix for baseline i-j.

    Note: has XX XY YX YY ordering

    :param gXi,gYi,dXYi,dYXi: current estimates of the complex gain and leakage
        terms for the first antenna in the baseline
    :param gXj,gYj,dXYj,dYXj: current estimates of the complex gain and leakage
        terms for the second antenna in the baseline
    :return: complex np array containing the 4x4 gain matrix
    """

    M = [
        [
            +gXi * conj(gXj),
            +gXi * conj(gXj * dXYj),
            +gXi * dXYi * conj(gXj),
            +gXi * dXYi * conj(gXj * dXYj),
        ],
        [
            -gXi * conj(gYj * dYXj),
            +gXi * conj(gYj),
            -gXi * dXYi * conj(gYj * dYXj),
            +gXi * dXYi * conj(gYj),
        ],
        [
            -gYi * dYXi * conj(gXj),
            -gYi * dYXi * conj(gXj * dXYj),
            +gYi * conj(gXj),
            +gYi * conj(gXj * dXYj),
        ],
        [
            +gYi * dYXi * conj(gYj * dYXj),
            -gYi * dYXi * conj(gYj),
            -gYi * conj(gYj * dYXj),
            +gYi * conj(gYj),
        ],
    ]

    return np.array(M)

def gen_coherency_products(Som, Smm, vobs, vmdl, wgt, ant, ant1, ant2):
    """
    Generate the 2x2 accumulations of coherency matrix products for antenna
    ant. Using a boolean mask for each antenna and expanding all of the matrix
    multiplication operations is faster than looping over baselines. 

    :param Som: accumulation matrices for observed vis multiplied by the
        Hermitian transpose of model vis [nant, nfreq, 2, 2]
    :param Smm: accumulation matrices for model vis multiplied by the
        Hermitian transpose of model vis [nant, nfreq, 2, 2]
    :param vobs: observed vis coherency matrices [ntime,nbaseline,nfreq,2,2]
    :param vmdl: model vis coherency matrices [ntime,nbaseline,nfreq,2,2]
    :param ant: antenna to accumulate
    :param ant1: first antenna in each baseline
    :param ant2: second antenna in each baseline
    """

    # update np.sums for station 1
    ind = ant1 == ant  # all baselines with ant as the first antenna
    ind *= ant2 != ant  # make sure they are cross-correlations
    Som[0, 0] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vobs[:, ind, :, 0, 0]
            * vmdl[:, ind, :, 0, 0].conj()
            + vobs[:, ind, :, 0, 1]
            * vmdl[:, ind, :, 0, 1].conj()
        )
    )
    Som[0, 1] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vobs[:, ind, :, 0, 0]
            * vmdl[:, ind, :, 1, 0].conj()
            + vobs[:, ind, :, 0, 1]
            * vmdl[:, ind, :, 1, 1].conj()
        )
    )
    Som[1, 0] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vobs[:, ind, :, 1, 0]
            * vmdl[:, ind, :, 0, 0].conj()
            + vobs[:, ind, :, 1, 1]
            * vmdl[:, ind, :, 0, 1].conj()
        )
    )
    Som[1, 1] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vobs[:, ind, :, 1, 0]
            * vmdl[:, ind, :, 1, 0].conj()
            + vobs[:, ind, :, 1, 1]
            * vmdl[:, ind, :, 1, 1].conj()
        )
    )
    #
    Smm[0, 0] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vmdl[:, ind, :, 0, 0]
            * vmdl[:, ind, :, 0, 0].conj()
            + vmdl[:, ind, :, 0, 1]
            * vmdl[:, ind, :, 0, 1].conj()
        )
    )
    Smm[0, 1] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vmdl[:, ind, :, 0, 0]
            * vmdl[:, ind, :, 1, 0].conj()
            + vmdl[:, ind, :, 0, 1]
            * vmdl[:, ind, :, 1, 1].conj()
        )
    )
    Smm[1, 0] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vmdl[:, ind, :, 1, 0]
            * vmdl[:, ind, :, 0, 0].conj()
            + vmdl[:, ind, :, 1, 1]
            * vmdl[:, ind, :, 0, 1].conj()
        )
    )
    Smm[1, 1] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vmdl[:, ind, :, 1, 0]
            * vmdl[:, ind, :, 1, 0].conj()
            + vmdl[:, ind, :, 1, 1]
            * vmdl[:, ind, :, 1, 1].conj()
        )
    )
    # update np.sums for station 2
    ind = ant2 == ant  # all baselines with ant as the second antenna
    ind *= ant1 != ant  # make sure they are cross-correlations
    Som[0, 0] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vobs[:, ind, :, 0, 0].conj()
            * vmdl[:, ind, :, 0, 0]
            + vobs[:, ind, :, 1, 0].conj()
            * vmdl[:, ind, :, 1, 0]
        )
    )
    Som[0, 1] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vobs[:, ind, :, 0, 0].conj()
            * vmdl[:, ind, :, 0, 1]
            + vobs[:, ind, :, 1, 0].conj()
            * vmdl[:, ind, :, 1, 1]
        )
    )
    Som[1, 0] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vobs[:, ind, :, 0, 1].conj()
            * vmdl[:, ind, :, 0, 0]
            + vobs[:, ind, :, 1, 1].conj()
            * vmdl[:, ind, :, 1, 0]
        )
    )
    Som[1, 1] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vobs[:, ind, :, 0, 1].conj()
            * vmdl[:, ind, :, 0, 1]
            + vobs[:, ind, :, 1, 1].conj()
            * vmdl[:, ind, :, 1, 1]
        )
    )
    #
    Smm[0, 0] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vmdl[:, ind, :, 0, 0].conj()
            * vmdl[:, ind, :, 0, 0]
            + vmdl[:, ind, :, 1, 0].conj()
            * vmdl[:, ind, :, 1, 0]
        )
    )
    Smm[0, 1] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vmdl[:, ind, :, 0, 0].conj()
            * vmdl[:, ind, :, 0, 1]
            + vmdl[:, ind, :, 1, 0].conj()
            * vmdl[:, ind, :, 1, 1]
        )
    )
    Smm[1, 0] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vmdl[:, ind, :, 0, 1].conj()
            * vmdl[:, ind, :, 0, 0]
            + vmdl[:, ind, :, 1, 1].conj()
            * vmdl[:, ind, :, 1, 0]
        )
    )
    Smm[1, 1] += np.sum(
        wgt[:, ind, :, 0]
        * (
            vmdl[:, ind, :, 0, 1].conj()
            * vmdl[:, ind, :, 0, 1]
            + vmdl[:, ind, :, 1, 1].conj()
            * vmdl[:, ind, :, 1, 1]
        )
    )
