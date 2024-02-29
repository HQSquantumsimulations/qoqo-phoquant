"""Python functions to compute vibronic spectra."""

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
from qoqo_strawberry_fields import StrawberryFieldsBackend
from qoqo import operations, Circuit
from typing import Tuple


def energy_for_samples(
    samples: np.array, freq_ini: np.array, freq_fin: np.array, E_ex: float = 0
) -> np.array:
    """Convert GBS samples to energies using molecular data.

    Args:
        samples: GBS samples
        freq_ini: vibrational frequencies of the initial state
        freq_fin: vibrational frequencies of the final state
        E_ex: vertical excitation energy

    Returns:
        numpy array: energies corresponding to the samples;
        one value per sample
    """
    # First, we compute ZPEs and add to excitation energy
    zpe_ini = 0.5 * np.sum(freq_ini)
    zpe_fin = 0.5 * np.sum(freq_fin)
    E_ex = E_ex + zpe_fin - zpe_ini
    # Compute energies from GBS samples: for zero T
    energies = []
    for sample in samples:
        energies.append(
            np.dot(sample, freq_fin.T))
    # Add zero-point-corrected excitation energy
    energies = np.array(energies)
    energies = energies + E_ex

    return energies


def mol_GBS(squeezing: np.array, shifts: np.array,
            ops1: list, ops2: list, shots: int)\
            -> Tuple[Circuit, np.array]:
    """GBS for vibronic molecular type of input using qoqo.

    Args:
        squeezing: squeezing parameters for the modes
        shifts: phase shift parameters for the modes
        ops1: list of qoqo operations for the first interferometer
        ops2: list of qoqo operations for the second interferometer
        shots: number of shots and measurements

    Returns:
        Tuple[Circuit, np.array]: qoqo circuit and measurements

    """

    # Extract number of modes
    nmodes = shifts.shape[0]

    # Create circuit
    circuit = Circuit()
    circuit += operations.DefinitionFloat("ro", nmodes, True)

    # Interferomener 1
    for op in ops1:
        circuit += op

    # Squeezing
    squeezing_angle = 0
    for mode in range(nmodes):
        circuit += operations.Squeezing(mode, squeezing[mode], squeezing_angle)

    # Interferometer 2
    for op in ops2:
        circuit += op

    # Phase displacement
    for mode in range(nmodes):
        circuit += operations.PhaseDisplacement(mode,
                                                np.abs(shifts[mode]),
                                                np.angle(shifts[mode]))

    # Measure
    for mode in range(nmodes):
        circuit += operations.PhotonDetection(mode, "ro", mode)

    circuit += operations.PragmaSetNumberOfMeasurements(shots, "ro")

    backend = StrawberryFieldsBackend(number_modes=3, device="gaussian")
    result = backend.run_circuit(circuit)

    # Post-process results
    samples = np.array(result[1]['ro'])

    return circuit, samples
