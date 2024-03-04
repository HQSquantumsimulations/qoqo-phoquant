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


def energy_for_samples(
    samples: np.ndarray, freq_ini: np.ndarray, freq_fin: np.ndarray, E_ex: float = 0
) -> np.ndarray:
    """Convert GBS samples to energies using molecular data.

    Args:
        samples: GBS samples
        freq_ini: vibrational frequencies of the initial state
        freq_fin: vibrational frequencies of the final state
        E_ex: vertical excitation energy

    Returns:
        ndarray: energies corresponding to the samples;
        one value per sample
    """
    # First, we compute ZPEs and add to excitation energy
    zpe_ini = 0.5 * np.sum(freq_ini)
    zpe_fin = 0.5 * np.sum(freq_fin)
    E_ex = E_ex + zpe_fin - zpe_ini
    # Compute energies from GBS samples: for zero T
    ener_samples = []
    for sample in samples:
        ener_samples.append(np.dot(sample, freq_fin.T))
    # Add zero-point-corrected excitation energy
    energies = np.array(ener_samples)
    energies = energies + E_ex

    return energies
