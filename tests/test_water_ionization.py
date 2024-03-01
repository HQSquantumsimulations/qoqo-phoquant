"""Tests of vibronic spectra calculations with GBS."""

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
import math
# qoqo_phoquant functionality
from qoqo_phoquant import unitary_to_ops
from qoqo_phoquant import molecule as mol
from qoqo_phoquant import spectrum_helpers as sh


def test_water_ionization() -> np.array:
    """Compute photoionization vibronic spectrum for water molecule.

    Returns:
        numpy array: energies corresponding to the samples;
        one value per sample
    """
    # Make molecule
    h2o = mol('../src/data/H2O_ion.json')

    # Decompose interferometers
    ops1 = unitary_to_ops(h2o.U1)
    ops2 = unitary_to_ops(h2o.U1)

    # Sample
    nshots = 1000
    cir, sam = sh.mol_GBS(squeezing=h2o.s, displ=h2o.alpha,
                          ops1=ops1, ops2=ops2, shots=nshots)

    # Convert to energies
    ener = sh.energy_for_samples(sam, h2o.freq_ini,
                                 h2o.freq_fin, h2o.E_vertical)

    sh.save_circuit(circuit=cir, name='H2O_ion_cir')

    # Checks
    errors = []
    # 1) Shape of sampling array
    ref_sam_shape = (nshots, len(h2o.freq_ini))
    sample_shape_correct = (ref_sam_shape == sam.shape)
    if not sample_shape_correct:
        errors.append("Wrong number of GBS shots or vibrations.")

    # 2) Shape of energies array
    ref_ener_shape = (nshots,)
    ener_shape_correct = (ref_ener_shape == ener.shape)
    if not ener_shape_correct:
        errors.append("Wrong number of sampled energies.")

    # 3) Minimum physically meaningful transition energy
    zpe_ini = 0.5 * np.sum(h2o.freq_ini)
    zpe_fin = 0.5 * np.sum(h2o.freq_fin)
    ref_min_E_ex = h2o.E_vertical + zpe_fin - zpe_ini
    min_ener_correct = math.isclose(ref_min_E_ex, min(ener), rel_tol=0.001)
    if not min_ener_correct:
        errors.append("Minimum excitation energy is wrong.")

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
