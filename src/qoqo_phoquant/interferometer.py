"""Python functions for interferometer unitary decomposition into qoqo operations."""

# Copyright © 2024 HQS Quantum Simulations GmbH. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#  distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express
# or implied. See the License for the specific language governing permissions
#  and limitations under
# the License.

import numpy as np
from typing import Tuple, List, Union, Any
from qoqo import operations


def T(m: int, n: int, theta: float, phi: float, nmax: int) -> np.array:
    """The Clements T matrix.

    Args:
            m: first index
            n: second index
            theta: transmittivity angle of beam splitter
            phi: phase angle of beam splitter
            nmax: square matrix dimension

    Returns:
            numpy array: Clements T matrix
    """
    t_mat = np.identity(nmax, dtype=np.complex128)
    t_mat[m, m] = np.exp(1j * phi) * np.cos(theta)
    t_mat[m, n] = -np.sin(theta)
    t_mat[n, m] = np.exp(1j * phi) * np.sin(theta)
    t_mat[n, n] = np.cos(theta)

    return t_mat


def T_inv(m: int, n: int, theta: float, phi: float, nmax: int) -> np.array:
    """The inverse of the Clements T matrix.

    Args:
            m: first index.
            n: second index.
            theta: transmittivity angle of beam splitter
            phi: phase angle of beam splitter
            nmax: square matrix dimension

    Returns:
            numpy array: inverse Clements T matrix
    """
    return np.transpose(T(m, n, theta, -phi, nmax))


def T_null(m: int, n: int, U: np.array) -> List[Union[float, int]]:
    """Parameters of the T matrix to nullify U[m, n].

    Args:
        m: first index
        n: second index
        U: unitary matrix

    Returns:
        List[Union[float, int]]: parameters for T function

    Raises:
        ValueError: U matrix is not square
    """
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("U is not a square matrix")

    if U[m, n] == 0:
        # Nothing here
        theta = 0
        phi = 0
    elif U[m-1, n] == 0:
        # Swap in the divide-by-zero case
        theta = np.pi / 2
        phi = 0
    else:
        # Non-trivial case
        r = -U[m, n] / U[m-1, n]
        theta = np.arctan(np.abs(r))
        phi = np.angle(r)

    return [m-1, m, theta, phi, nmax]


def inv_T_null(m: int, n: int, U: np.array) -> List[Union[float, int]]:
    """Parameters of the inverse T matrix to nullify U[m, n].

    Args:
        m: first index
        n: second index
        U: unitary matrix

    Returns:
        List[Union[float, int]]: parameters for T_inv function

    Raises:
        ValueError: U matrix is not square
    """
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("U must be a square matrix")

    if U[m, n] == 0:
        # No swaps for the identity-like case
        theta = 0
        phi = 0
    elif U[m, n + 1] == 0:
        # Swap in the divide-by-zero case
        theta = np.pi / 2
        phi = 0
    else:
        r = U[m, n] / U[m, n + 1]
        theta = np.arctan(np.abs(r))
        phi = np.angle(r)

    return [n, n + 1, theta, phi, nmax]


def interf_rect_decompose(U: np.array) -> Tuple[
     List[np.array], np.array, List[np.array]]:
    """Recatangular (Clement's) decomposition of the interferometer.

    Args:
        U: interferometer's unitary matrix

    Returns:
        Tuple[List[np.array], np.array, List[np.array]]: matrices for
            Clement's decomposition

    Raises:
        ValueError: The interferometer matrix is not unitary
    """
    # Check whether U is unitary
    mat = np.matrix(U)
    if not np.allclose(np.eye(mat.shape[0]), mat.H @ mat):
        raise ValueError("The interferometer matrix is not unitary")
    else:
        print("The interferometer matrix is unitary")

    n_channels = mat.shape[0]
    print(n_channels)
    print(mat)

    # Initiate two-channel transformations
    # Lists with parameters
    T_list = []
    T_inv_list = []
    for k, i in enumerate(range(n_channels - 2, - 1, - 1)):
        # Even
        if k % 2 == 0:
            for j in reversed(range(n_channels - 1 - i)):
                # Find nullyfying T_inv
                t_inv = inv_T_null(i + j + 1, j, mat)
                # Update U
                mat = mat @ T_inv(*t_inv)
                # Update list of inverse T
                T_inv_list.append(t_inv)
        else:
            for j in range(n_channels - 1 - i):
                # Find nullyfuing T
                t = T_null(i + j + 1, j, mat)
                # Update U
                mat = T(*t) @ mat
                T_list.append(t)

    return T_inv_list, np.diag(mat), T_list


def unitary_to_ops(U: np.array) -> List["operations.Operation"]:
    """Converts interferometer's unitary into list of qoqo operations.

    Args:
        U: interferometer's unitary matrix

    Returns:
        List[operations.Operation]: list of qoqo operations
    """
    T_i, diag, T = interf_rect_decompose(U)

    #
    # Convert numbers to operations
    #
    decomposition_thresh = 1.0e-13

    # T_i
    ops: List[operations.Operation] = []

    for n, m, theta_raw, phi_raw, _ in T_i:
        theta = theta_raw if np.abs(theta_raw) >= decomposition_thresh else 0
        phi = phi_raw if np.abs(phi_raw) >= decomposition_thresh else 0

        if phi != 0:
            # Phase shift
            print(operations.PhaseShift(n, phi))
            ops.append(operations.PhaseShift(n, phi))
        if theta != 0:
            # Beam splitter
            print(operations.BeamSplitter(n, m, theta, 0))
            ops.append(operations.BeamSplitter(n, m, theta, 0))

    # Diagonal part
    for n, expphi in enumerate(diag):
        # Local phase shifts

        if np.abs(expphi - 1) >= decomposition_thresh:
            q = np.log(expphi).imag
        else:
            q = 0
        if (q != 0):
            print(operations.PhaseShift(n, np.mod(q, 2*np.pi)))
            ops.append(operations.PhaseShift(n, np.mod(q, 2*np.pi)))

    # T
    for n, m, theta_raw, phi_raw, _ in reversed(T):
        theta = theta_raw if np.abs(theta_raw) >= decomposition_thresh else 0
        phi = phi_raw if np.abs(phi_raw) >= decomposition_thresh else 0

        if theta != 0:
            # Beam Splitter
            print(operations.BeamSplitter(n, m, -theta, 0))
            ops.append(operations.BeamSplitter(n, m, -theta, 0))

        if phi != 0:
            # Phase shift
            print(operations.PhaseShift(n, -phi))
            ops.append(operations.PhaseShift(n, -phi))

    return ops
