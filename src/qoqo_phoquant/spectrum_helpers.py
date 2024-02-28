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


def energy_for_samles(samples: np.array, freq_ini: np.array,
                      freq_fin: np.array, E_ex: float = 0) -> np.array:
    """Convert GBS samples to energies using molecular data.

    Args:
        samples: GBS samples
        freq_ini: vibrational frequencies of the initial state
        freq_fin: vibrational frequencies of the final state
        E_ex: vertical excitation energy

    Returns:
        numpy array: energies corresponding to the samples
    """
    # First, we compute ZPEs and add to excitation energy
    zpe_ini = 0.5*np.sum(freq_ini)
    zpe_fin = 0.5*np.sum(freq_fin)
    E_ex = E_ex + zpe_fin - zpe_ini
    # Compute energies from GBS samples
    energies = []
    for sample in samples:
        energies.append(np.dot(sample[: len(sample) // 2], freq_fin)
                        - np.dot(sample[len(sample) // 2:], freq_ini))
    # Add zero-point-corrected excitation energy
    energies = np.array(energies)
    energies = energies + E_ex

    return energies
