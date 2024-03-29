"""Wrappers to compute vibronic spectra using qoqo-strawberry-fields."""

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
from .interferometer import unitary_to_ops
from .molecules import molecule
from .spectrum_helpers import energy_for_samples, mol_GBS, save_circuit
